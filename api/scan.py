import io
import math
import os
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageOps

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

APP_NAME = "scan-endpoint"
TIMEOUT = httpx.Timeout(30.0, read=60.0)

app = FastAPI(title=APP_NAME)


# ------------------------------
# Networking
# ------------------------------
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

async def download_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": UA}) as client:
        r = await client.get(url, follow_redirects=True)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
    img = Image.open(buf)
    # wandeln in RGB (einige Airtable-Uploads sind HEIC/JPEG/PNG)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


# ------------------------------
# Bildverarbeitung
# ------------------------------
def pil_to_np_gray(pil: Image.Image) -> np.ndarray:
    if pil.mode != "L":
        pil = pil.convert("L")
    return np.array(pil, dtype=np.uint8)

def auto_rotate_by_min_area(gray: np.ndarray) -> np.ndarray:
    # grobe Rotation aufrecht; nutzt größten Konturwinkel
    edges = cv2.Canny(gray, 60, 180)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    # OpenCV-Winkel-Interpretation anpassen
    if angle < -45:
        angle = 90 + angle
    # nur drehen, wenn relevant
    if abs(angle) < 1.5:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def find_receipt_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    # Kanten → größte 4‑Punkt-Kontur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 180)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2).astype(np.float32)
    return None

def order_quad(pts: np.ndarray) -> np.ndarray:
    # sortiert 4 Punkte (tl, tr, br, bl)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

def warp_to_rect(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    quad = order_quad(quad)
    (tl, tr, br, bl) = quad
    # Zielgröße aus Seitenlängen mitteln
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    W = int(max(widthA, widthB))
    H = int(max(heightA, heightB))
    # sinnvolle Obergrenze
    scale = 1.0
    max_side = 2200
    if max(W,H) > max_side:
        scale = max_side / max(W,H)
        W = int(W*scale)
        H = int(H*scale)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(gray, M, (W,H), flags=cv2.INTER_CUBIC)

def binarize(gray: np.ndarray) -> np.ndarray:
    # lokale Kontrastanhebung
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    # Rauschen schonend dämpfen
    smooth = cv2.bilateralFilter(cl, d=7, sigmaColor=60, sigmaSpace=60)
    # adaptives Threshold
    bw = cv2.adaptiveThreshold(
        smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    # kleine Artefakte tilgen
    bw = cv2.medianBlur(bw, 3)
    return bw

def preprocess(pil: Image.Image) -> np.ndarray:
    # in Graustufen + leichtes Downscale falls RIESIG
    base = pil
    if max(base.size) > 3000:
        scale = 3000 / max(base.size)
        base = base.resize((int(base.width*scale), int(base.height*scale)), Image.LANCZOS)

    gray = pil_to_np_gray(base)
    gray = auto_rotate_by_min_area(gray)

    quad = find_receipt_quad(gray)
    if quad is not None:
        gray = warp_to_rect(gray, quad)

    # Binärbild
    bw = binarize(gray)
    # sicherstellen: 0/255, uint8
    bw = (bw > 127).astype(np.uint8) * 255
    return bw


# ------------------------------
# PDF-Erzeugung
# ------------------------------
def make_pdf_from_bw(bw: np.ndarray) -> bytes:
    # schwarz = Text (0), weiß = Hintergrund (255)
    # Bild nach PIL
    pil = Image.fromarray(bw, mode="L")
    # für ReportLab als ImageReader
    img_reader = ImageReader(pil)

    page_w, page_h = A4  # Punkte (1/72")
    margin = 36  # 0.5 inch
    avail_w = page_w - 2*margin
    avail_h = page_h - 2*margin

    img_w, img_h = pil.size
    # Seitenverhältnis sichern
    scale = min(avail_w / img_w, avail_h / img_h)
    out_w = img_w * scale
    out_h = img_h * scale
    x = (page_w - out_w) / 2
    y = (page_h - out_h) / 2

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    # Weißer Hintergrund
    c.setFillColorRGB(1,1,1)
    c.rect(0,0,page_w,page_h, stroke=0, fill=1)
    # Bild
    c.drawImage(img_reader, x, y, width=out_w, height=out_h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    return buf.getvalue()


# ------------------------------
# Upload (Failover)
# ------------------------------
async def upload_tmpfiles(pdf_bytes: bytes, filename: str) -> Optional[str]:
    # https://tmpfiles.org/ – multipart/form-data
    data = {"file": (filename, pdf_bytes, "application/pdf")}
    async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": UA}) as client:
        r = await client.post("https://tmpfiles.org/api/v1/upload", files=data)
        if r.status_code == 200:
            js = r.json()
            # API liefert „data.url“, Download ist dort direkt verlinkt
            url = js.get("data", {}).get("url")
            if url:
                # Auf der Seite steht der unmittelbare pdf-Link; üblich ist /dl/<id>/filename
                # wir versuchen denselben Pfad – viele UIs zeigen ihn als einzigen Link
                return url
    return None

async def upload_0x0(pdf_bytes: bytes, filename: str) -> Optional[str]:
    # https://0x0.st – multipart: field = 'file'
    data = {"file": (filename, pdf_bytes, "application/pdf")}
    async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": UA}) as client:
        r = await client.post("https://0x0.st", files=data)
        if r.status_code == 200 and r.text.startswith("https://"):
            return r.text.strip()
    return None

async def upload_with_failover(pdf_bytes: bytes, filename: str) -> str:
    # 1) tmpfiles → 2) 0x0
    for fn in (upload_tmpfiles, upload_0x0):
        try:
            link = await fn(pdf_bytes, filename)
            if link:
                return link
        except Exception:
            continue
    raise RuntimeError("Kein Upload-Endpunkt erreichbar")


# ------------------------------
# API
# ------------------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"

@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Direkter Bild-URL (Airtable Attachment URL)"),
    response: str = Query("link", pattern="^(link|pdf|base64)$")
):
    # 1) Bild laden
    img = await download_image(file_url)

    # 2) Vorverarbeitung
    bw = preprocess(img)

    # 3) PDF bauen
    pdf_bytes = make_pdf_from_bw(bw)
    filename = "scan.pdf"

    # 4) Antwort
    if response == "pdf":
        # für Zapier „Custom Request“ raw body
        return PlainTextResponse(pdf_bytes, media_type="application/pdf")
    if response == "base64":
        import base64
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        return JSONResponse({"filename": filename, "content_type": "application/pdf", "file_base64": b64})

    # default: link
    link = await upload_with_failover(pdf_bytes, filename)
    return JSONResponse({"link": link, "note": "Link ist temporär. Bitte zeitnah von Airtable abrufen."})
