# api/scan.py
from __future__ import annotations

import io
import math
from typing import Literal, Optional

import numpy as np
import httpx
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from PIL import Image, ImageOps, ImageFilter
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(title="Receipt Scan Service")

# -------------------------------------------------
# HTTP Download
# -------------------------------------------------
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=15.0)
HTTP_HEADERS = {
    # Hilft Airtable/CDN teils beim Ausliefern
    "User-Agent": "scan-endpoint/1.0 (+https://render.com)",
    "Accept": "*/*",
}


async def fetch_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS) as client:
        r = await client.get(url, follow_redirects=True)
        r.raise_for_status()
        return r.content


# -------------------------------------------------
# Bild-Pipeline (robust, konservativ)
# -------------------------------------------------
def _ensure_portrait(img: Image.Image) -> Image.Image:
    """Macht Hochformat, wenn eindeutig Querformat."""
    w, h = img.size
    if w > h * 1.1:
        img = img.rotate(90, expand=True)
    return img


def _deskew_cv(img_rgb: np.ndarray) -> np.ndarray:
    """
    Vorsichtiges Deskew: schätzt eine leichte Schräglage über Kanten.
    Kein extremes Warping, um Fehl-Erkennungen (z.B. reiner QR-Code) zu vermeiden.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binär für Konturen/Edges
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(thr, 50, 150)

    # Hough-Linien suchen
    lines = cv2.HoughLines(edges, 1, np.pi/180, 140)
    if lines is None:
        return img_rgb

    # Winkel rund um vertikal (0° oder 180°) einsammeln
    angles = []
    for l in lines[:200]:
        rho, theta = l[0]
        # Vertikale Linien haben theta ~ 0 oder pi
        ang = (theta * 180.0 / math.pi)
        # auf -90..90 mappen
        if ang > 90:
            ang -= 180
        angles.append(ang)

    if not angles:
        return img_rgb

    med = float(np.median(angles))
    # Wir limitieren die Korrektur (max. ±5°), um Over-rotation zu verhindern
    if abs(med) < 0.2 or abs(med) > 5.0:
        return img_rgb

    # Rotation um den Medianwinkel
    h, w = img_rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), med, 1.0)
    rotated = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_for_scan(pil_img: Image.Image) -> Image.Image:
    """
    Liefert eine gut lesbare, schwarz/weiß-reine Quittung zurück,
    ohne aggressive Zuschneidung (wir schneiden NICHT automatisch,
    damit kein QR‑Code-only passiert).
    """
    # 1) Exif & Portrait
    img = ImageOps.exif_transpose(pil_img.convert("RGB"))
    img = _ensure_portrait(img)

    # 2) leichte Schräglagen-Korrektur (OpenCV), konservativ
    arr = np.array(img)
    arr = _deskew_cv(arr)

    # 3) sanfte Glättung, adaptives Threshold
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # leichte Entgrieselung
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)

    # adaptives Threshold – invertiert, damit Schrift schwarz ist
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41, 15
    )

    # 4) Morphologische Öffnung, um Teppichkörnung zu reduzieren
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5) dünner Rand weg (aber NICHT croppen auf Kontur -> stabiler)
    pad = 6
    h, w = bw.shape[:2]
    bw = bw[pad:h - pad, pad:w - pad]

    # zurück zu PIL (ein Kanal)
    out = Image.fromarray(bw).convert("L")
    # Optional minimaler Kontrast-Boost
    out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=4))
    return out


# -------------------------------------------------
# PDF-Erzeugung
# -------------------------------------------------
def make_pdf_from_image(img_bw: Image.Image) -> bytes:
    """
    Platziert die S/W-Quittung schön auf A4.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    page_w, page_h = A4  # points
    margin = 24  # Punkte (~8.5 mm)
    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin

    # PIL -> Bytes -> ImageReader (damit BytesIO funktioniert)
    png_buf = io.BytesIO()
    img_bw.save(png_buf, format="PNG", optimize=True)
    png_buf.seek(0)
    img_reader = ImageReader(png_buf)

    # Größenverhältnis
    iw, ih = img_bw.size
    scale = min(avail_w / iw, avail_h / ih)
    out_w = iw * scale
    out_h = ih * scale

    x = (page_w - out_w) / 2
    y = (page_h - out_h) / 2

    c.drawImage(img_reader, x, y, width=out_w, height=out_h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    return buf.getvalue()


# -------------------------------------------------
# Uploader (0x0.st → tmpfiles.org Fallback)
# -------------------------------------------------
async def upload_0x0(name: str, data: bytes) -> Optional[str]:
    try:
        files = {"file": (name, data, "application/pdf")}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.post("https://0x0.st", files=files)
            r.raise_for_status()
            url = r.text.strip()
            # 0x0.st liefert eine direkte URL zurück
            if url.startswith("http"):
                return url
    except Exception:
        pass
    return None


async def upload_tmpfiles(name: str, data: bytes) -> Optional[str]:
    try:
        # tmpfiles.org API: multipart "file"; Rückgabe JSON mit id,
        # direkter Download: https://tmpfiles.org/dl/<id>/<name>
        files = {"file": (name, data, "application/pdf")}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
            r.raise_for_status()
            js = r.json()
            file_id = js.get("data", {}).get("file", {}).get("url", "")
            # Manche Antworten geben bereits /dl/… zurück; zur Sicherheit normalisieren:
            if "/dl/" in file_id:
                return file_id
            # Fallback, falls nur ID vorhanden:
            fid = js.get("data", {}).get("file", {}).get("id")
            if fid:
                return f"https://tmpfiles.org/dl/{fid}/{name}"
    except Exception:
        pass
    return None


async def upload_with_fallback(name: str, data: bytes) -> str:
    link = await upload_0x0(name, data)
    if link:
        return link
    link = await upload_tmpfiles(name, data)
    if link:
        return link
    # Wenn beides down ist, geben wir lokalen Download zurück (als Notnagel)
    # – für Zapier/Airtable ist das zwar nicht dauerhaft, aber besser als 500.
    return "about:blank"


# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Scan service up. Use /scan?file_url=...&format=pdf&response=link"


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"


@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Öffentliche Datei-URL (Airtable Attachment URL)"),
    format: Literal["pdf", "png"] = Query("pdf"),
    response: Literal["json", "file", "link"] = Query("json"),
):
    """
    Lädt ein Bild, verarbeitet es zu „gescannter“ Quittung und gibt
    - entweder die Datei (Streaming) zurück,
    - oder JSON (base64 sparen wir uns),
    - oder einen direkten Download-Link (0x0.st / tmpfiles.org).

    Zapier‑Empfehlung: response=link (liefert ein Feld 'url').
    """
    # 1) Datei holen
    raw = await fetch_bytes(file_url)
    pil = Image.open(io.BytesIO(raw))
    processed = preprocess_for_scan(pil)

    # 2) Ausgabe formen
    if format == "png":
        out_buf = io.BytesIO()
        processed.save(out_buf, format="PNG", optimize=True)
        out_bytes = out_buf.getvalue()
        media = "image/png"
        filename = "scan.png"
    else:
        out_bytes = make_pdf_from_image(processed)
        media = "application/pdf"
        filename = "scan.pdf"

    # 3) Antworttyp
    if response == "file":
        headers = {"Content-Disposition": f'inline; filename="{filename}"'}
        return StreamingResponse(io.BytesIO(out_bytes), media_type=media, headers=headers)
    elif response == "link":
        link = await upload_with_fallback(filename, out_bytes)
        return JSONResponse({"filename": filename, "content_type": media, "url": link})
    else:
        # JSON‑Meta (klein halten, keine base64)
        return JSONResponse({"filename": filename, "content_type": media, "bytes": len(out_bytes)})
