# api/scan.py
import io
import math
import uuid
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from PIL import Image, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Scan Endpoint")

UA = "scan-endpoint/1.0 (+https://example.invalid)"
TIMEOUT = httpx.Timeout(30.0, read=60.0)


# -----------------------------
# Bild-Preprocessing (robuster)
# -----------------------------
def to_rgb(pil: Image.Image) -> Image.Image:
    if pil.mode == "RGBA":
        # auf weiß setzen statt schwarz
        bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        bg.alpha_composite(pil)
        pil = bg.convert("RGB")
    elif pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil


def estimate_deskew_angle(gray: np.ndarray) -> float:
    # Hough-basierte Schätzung
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
    if lines is None:
        return 0.0
    angles = []
    for rho, theta in lines[:, 0]:
        # nahe waagrecht/senkrecht
        a = (theta % (np.pi / 2.0))
        a = min(a, abs(np.pi / 2.0 - a))
        deg = (theta * 180.0 / np.pi)
        if a < np.deg2rad(15):
            # auf -45..+45 bringen
            d = ((deg + 45) % 90) - 45
            angles.append(d)
    if not angles:
        return 0.0
    return float(np.median(angles))


def preprocess(pil_in: Image.Image) -> Image.Image:
    import cv2  # lazy import, damit requirements klar sind

    pil = to_rgb(pil_in)
    # große Bilder vorher auf ~12MP limitieren (Performanz)
    max_side = 4000
    if max(pil.size) > max_side:
        pil = ImageOps.contain(pil, (max_side, max_side), Image.LANCZOS)

    # leichte Kontrastanhebung
    pil = ImageOps.autocontrast(pil, cutoff=1)

    arr = np.asarray(pil)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Deskew
    try:
        angle = estimate_deskew_angle(gray)
        if abs(angle) > 0.2:
            h, w = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(
                gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
    except Exception:
        pass

    # adaptives Thresholding -> binär
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )

    # leichtes Öffnen, um Teppich/Noise zu killen
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    # Randpolster wegschneiden (robust gegen Größenfehler)
    pad = 3
    H, W = bw.shape[:2]
    y0, y1 = pad, max(0, H - pad)
    x0, x1 = pad, max(0, W - pad)
    bw = bw[y0:y1, x0:x1]

    # zurück zu PIL (als „weißer Hintergrund“)
    pil_bw = Image.fromarray(bw).convert("L")
    return pil_bw


# -----------------------------
# PDF-Erstellung
# -----------------------------
def make_pdf_from_image(img: Image.Image) -> bytes:
    # Bild (L oder RGB) in PNG-Bytes konvertieren
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    # A4 PDF, Seite im Hochformat
    page_w, page_h = A4
    cbuf = io.BytesIO()
    c = canvas.Canvas(cbuf, pagesize=A4)

    png_reader = ImageReader(buf)
    # Bildgröße in Points schätzen (dpi ~ 300)
    # Wir skalieren proportional auf A4 mit Rändern
    margin = 36  # 0.5 inch
    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin

    # tatsächliche Pixelgröße ermitteln
    pil = Image.open(buf)
    w_px, h_px = pil.size
    # Seitenverhältnis
    r_img = w_px / float(h_px)
    r_page = avail_w / float(avail_h)

    if r_img >= r_page:
        out_w = avail_w
        out_h = avail_w / r_img
    else:
        out_h = avail_h
        out_w = avail_h * r_img

    x = (page_w - out_w) / 2.0
    y = (page_h - out_h) / 2.0

    c.drawImage(png_reader, x, y, width=out_w, height=out_h, mask="auto")
    c.showPage()
    c.save()
    pdf_bytes = cbuf.getvalue()
    cbuf.close()
    return pdf_bytes


# -----------------------------
# Uploader (mit Fallbacks)
# -----------------------------
async def upload_0x0(pdf_bytes: bytes, filename: str) -> Optional[str]:
    # https://0x0.st  – liefert direkten Link im Klartext
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    headers = {"User-Agent": UA}
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers) as client:
        r = await client.post("https://0x0.st", files=files)
        if r.status_code == 200 and r.text.startswith("https://"):
            return r.text.strip()
    return None


async def upload_fileio(pdf_bytes: bytes, filename: str) -> Optional[str]:
    # https://www.file.io – JSON mit "link"
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    headers = {"User-Agent": UA}
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers) as client:
        r = await client.post("https://file.io", files=files)
        if r.status_code == 200:
            try:
                data = r.json()
                if data.get("success") and data.get("link"):
                    return str(data["link"])
            except Exception:
                pass
    return None


async def upload_tmpfiles(pdf_bytes: bytes, filename: str) -> Optional[str]:
    # https://tmpfiles.org – gibt Seiten-URL zurück; direkter Link = /dl/<id>/<filename>
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    headers = {"User-Agent": UA}
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers) as client:
        r = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
        # Alternative (ältere Variante): POST / – dann HTML parsen; hier bevorzugt API.
        if r.status_code in (200, 201):
            try:
                data = r.json()
                url = data.get("data", {}).get("url")
                if url and "/file/" in url:
                    # API liefert /file/<id> – direkter Download ist /dl/<id>/<filename>
                    file_id = url.rstrip("/").split("/")[-1]
                    return f"https://tmpfiles.org/dl/{file_id}/{filename}"
            except Exception:
                pass
    return None


async def upload_with_fallbacks(pdf_bytes: bytes, filename: str) -> str:
    for fn in (upload_0x0, upload_fileio, upload_tmpfiles):
        try:
            link = await fn(pdf_bytes, filename)
            if link:
                return link
        except Exception:
            continue
    # Wenn alles schiefgeht, geben wir eine leere Zeichenkette zurück
    return ""


# -----------------------------
# Download & Routing
# -----------------------------
async def fetch_bytes(url: str) -> tuple[bytes, str]:
    headers = {"User-Agent": UA}
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "").lower()


def looks_like_pdf(content_type: str, first_bytes: bytes) -> bool:
    if "application/pdf" in content_type:
        return True
    return first_bytes[:4] == b"%PDF"


@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"


@app.get("/scan", response_class=PlainTextResponse)
async def scan(file_url: str = Query(..., description="Öffentliche URL des Bildes oder PDFs")):
    # 1) Datei laden
    raw, ctype = await fetch_bytes(file_url)

    # 2) Wenn schon PDF -> hochladen
    if looks_like_pdf(ctype, raw):
        pdf_bytes = raw
    else:
        # Bild einlesen
        img = Image.open(io.BytesIO(raw))
        # Hochformat erzwingen (lange Kante vertikal, falls nötig)
        if img.width > img.height:
            img = img.rotate(90, expand=True)
        bw = preprocess(img)  # PIL (L)
        pdf_bytes = make_pdf_from_image(bw)

    # 3) Hochladen (mit Fallbacks)
    filename = f"scan_{uuid.uuid4().hex[:8]}.pdf"
    link = await upload_with_fallbacks(pdf_bytes, filename)

    if not link:
        # als Notnagel: PDF als Data-URL (Airtable kann das NICHT direkt brauchen,
        # aber so ist ein minimaler Rückkanal vorhanden)
        return PlainTextResponse("ERROR: upload failed", status_code=502)

    # 4) Nur den Link als Plain-Text zurückgeben
    return PlainTextResponse(link, media_type="text/plain")
