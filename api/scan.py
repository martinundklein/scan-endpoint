# api/scan.py
# Vollständige FastAPI-App für "Scannen" (deskew, crop, B/W) + PDF + Link-Upload mit Fallbacks

from __future__ import annotations

import io
import math
import os
from typing import Optional, Literal

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel, HttpUrl
import httpx

# Bildverarbeitung
import cv2  # opencv-python-headless
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.utils import ImageReader

app = FastAPI(title="scan-endpoint", version="1.0.0")

# ------------- Hilfsfunktionen: Download ----------------

async def fetch_bytes(url: str, timeout: float = 30.0) -> bytes:
    headers = {"User-Agent": "scan-endpoint/1.0"}
    to = httpx.Timeout(timeout, connect=10)
    async with httpx.AsyncClient(headers=headers, timeout=to, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code >= 400:
            raise HTTPException(status_code=400, detail=f"Download fehlgeschlagen ({r.status_code}).")
        return r.content

# ------------- Bildverarbeitung -------------------------

def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # basierend auf PyImageSearch-Standard
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def _detect_document_and_warp(bgr: np.ndarray) -> np.ndarray:
    ratio = bgr.shape[0] / 500.0
    image_small = cv2.resize(bgr, (int(bgr.shape[1] / ratio), 500))
    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = _auto_canny(gray)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2) * ratio
            return _four_point_transform(bgr, pts.astype("float32"))

    # Fallback: keine 4 Ecken gefunden -> Original zurück
    return bgr

def _deskew(gray: np.ndarray) -> np.ndarray:
    # einfache Schätzung über minAreaRect der größten Komponente
    bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bin_img = 255 - bin_img  # Text = weiß
    coords = np.column_stack(np.where(bin_img > 0))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _to_scanned_bw(bgr: np.ndarray) -> np.ndarray:
    # Perspektive -> Graustufen -> Deskew -> Kontrast/CLAHE -> Adaptive Threshold
    warped = _detect_document_and_warp(bgr)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15
    )
    return bw

def make_pdf_from_image(np_img_bw: np.ndarray, prefer_a4: bool = True) -> bytes:
    # Setze Seite (A4) und skaliere das Bild maximal darauf
    page_size = A4 if prefer_a4 else letter
    page_w, page_h = page_size

    pil_img = Image.fromarray(np_img_bw)
    # Invertiere ggf., damit Hintergrund weiß bleibt
    if np.mean(np_img_bw) < 127:
        pil_img = Image.fromarray(255 - np_img_bw)

    img_buf = io.BytesIO()
    pil_img.save(img_buf, format="PNG")
    img_buf.seek(0)

    pdf_bytes = io.BytesIO()
    c = canvas.Canvas(pdf_bytes, pagesize=page_size)

    img = ImageReader(img_buf)
    iw, ih = pil_img.size

    # DPI ~ 200 → Pixel zu Punkt (1 Punkt = 1/72 Zoll): 200 px ≈ 72 pt -> scale ~ 72/200
    scale = 72.0 / 200.0
    img_w_pt = iw * scale
    img_h_pt = ih * scale

    # Fit auf Seite mit Rand
    max_w = page_w - 36 * 2
    max_h = page_h - 36 * 2
    factor = min(max_w / img_w_pt, max_h / img_h_pt)
    draw_w = img_w_pt * factor
    draw_h = img_h_pt * factor
    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2

    c.drawImage(img, x, y, width=draw_w, height=draw_h, mask='auto')
    c.showPage()
    c.save()
    return pdf_bytes.getvalue()

# ------------- Upload mit Fallbacks ---------------------

async def _upload_transfer_sh(client: httpx.AsyncClient, pdf_bytes: bytes, filename: str) -> Optional[str]:
    try:
        r = await client.put(f"https://transfer.sh/{filename}", content=pdf_bytes)
        if r.status_code == 200 and r.text.strip().startswith("https://"):
            return r.text.strip()
    except Exception:
        pass
    return None

async def _upload_tmpfiles(client: httpx.AsyncClient, pdf_bytes: bytes, filename: str) -> Optional[str]:
    try:
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        r = await client.post("https://tmpfiles.org/api/v1/upload", files=files, timeout=30)
        if r.status_code == 200:
            data = r.json()
            url = (data or {}).get("data", {}).get("url")
            if isinstance(url, str) and url.startswith("http"):
                return url
    except Exception:
        pass
    return None

async def _upload_0x0(client: httpx.AsyncClient, pdf_bytes: bytes, filename: str) -> Optional[str]:
    try:
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        r = await client.post("https://0x0.st", files=files, timeout=30)
        if r.status_code == 200 and r.text.strip().startswith("http"):
            return r.text.strip()
    except Exception:
        pass
    return None

async def upload_with_fallbacks(pdf_bytes: bytes, filename: str) -> str:
    headers = {"User-Agent": "scan-endpoint/1.0"}
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
        for fn in (_upload_transfer_sh, _upload_tmpfiles, _upload_0x0):
            url = await fn(client, pdf_bytes, filename)
            if url:
                return url
    raise RuntimeError("Upload fehlgeschlagen (alle Provider).")

# ------------- API-Modelle --------------------------------

class ScanBody(BaseModel):
    file_url: HttpUrl
    response: Literal["link", "pdf"] = "link"

# ------------- Endpunkte ----------------------------------

@app.get("/", response_class=PlainTextResponse, include_in_schema=False)
async def root():
    return "scan-endpoint OK"

@app.get("/health", response_class=PlainTextResponse, include_in_schema=False)
async def health():
    return "healthy"

@app.get("/scan")
async def scan_get(
    file_url: HttpUrl = Query(..., description="Öffentliche URL zum Bild/PDF (Airtable Attachment URL)"),
    response: Literal["link", "pdf"] = Query("link", description="Antwort als Download-Link oder direktes PDF"),
):
    return await _scan_core(str(file_url), response)

@app.post("/scan")
async def scan_post(body: ScanBody):
    return await _scan_core(str(body.file_url), body.response)

# ------------- Kernlogik ----------------------------------

async def _scan_core(file_url: str, response_mode: Literal["link", "pdf"]):
    # 1) Download
    raw = await fetch_bytes(file_url)

    # 2) Lade als Bild
    try:
        npimg = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if bgr is None:
            # evtl. PDF? Dann erste Seite als Bild extrahieren (hier nicht implementiert)
            raise ValueError("Kein Bildformat erkannt.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Datei konnte nicht als Bild geladen werden: {e}")

    # 3) „Scannen“ (B/W, Beschnitt, Begradigen)
    bw = _to_scanned_bw(bgr)

    # 4) PDF erzeugen
    pdf_bytes = make_pdf_from_image(bw, prefer_a4=True)
    filename = "scan.pdf"

    # 5) Antwort
    if response_mode == "pdf":
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    # 6) Link hochladen (Fallbacks)
    try:
        url = await upload_with_fallbacks(pdf_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload fehlgeschlagen: {e}")

    return JSONResponse(
        {
            "filename": filename,
            "content_type": "application/pdf",
            "url": url,
        }
    )
