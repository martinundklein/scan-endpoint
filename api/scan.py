# api/scan.py
# FastAPI-Endpoint: /scan
# Features:
# - GET /scan?file_url=...&response=link|pdf
# - POST /scan  { "file_url": "...", "response": "link|pdf" }
# - Lädt Bild von URL, richtet aus (EXIF + Deskew), beschneidet, wandelt in
#   kontrastreiches S/W um ("gescannt"), erzeugt PDF.
# - response=link -> lädt PDF zu tmpfiles.org (API) hoch und gibt Direktlink zurück
# - response=pdf  -> sendet PDF-Bytes direkt (application/pdf)

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, HttpUrl
import io
import math
import numpy as np
import cv2
from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait
from reportlab.lib.utils import ImageReader
import httpx
import asyncio

app = FastAPI(title="Scan Endpoint", version="2.0.0")

# -----------------------------
# Helper: HTTP download
# -----------------------------
async def fetch_bytes(url: str, timeout_s: float = 20.0) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download fehlgeschlagen: {e}")

# -----------------------------
# Helper: PIL <-> OpenCV
# -----------------------------
def pil_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    # EXIF-Orientierung berücksichtigen
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im.convert("RGB")

def cv_from_pil(im: Image.Image) -> np.ndarray:
    arr = np.array(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def pil_from_cv(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# -----------------------------
# Bild-Verarbeitung
# -----------------------------
def order_points(pts):
    # sortiert 4 Punkte im Uhrzeigersinn: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
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
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    return warped

def largest_receipt_region(img_bgr: np.ndarray):
    # Sucht größte, annähernd rechteckige Kontur (Beleg)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Größte Kontur nach Fläche
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        box = approx.reshape(4, 2).astype("float32")
        return four_point_transform(img_bgr, box)

    # Fallback: Deskew via minAreaRect
    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    # Rotieren um Mittelpunkt
    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Dann Bounding-Box der größten Kontur auf dem rotierten Bild
    gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    edges2 = cv2.Canny(gray2, 40, 120)
    edges2 = cv2.dilate(edges2, np.ones((3, 3), np.uint8), iterations=1)
    cnts2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts2:
        cnt2 = max(cnts2, key=cv2.contourArea)
        x, y, w2, h2 = cv2.boundingRect(cnt2)
        cropped = rotated[max(y-10,0):y+h2+10, max(x-10,0):x+w2+10]
        return cropped

    return rotated

def to_scanned_bw(img_bgr: np.ndarray) -> np.ndarray:
    # In Graustufen, entrauschen, adaptiv thresholden -> "gescannt"
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # leicht glätten, Erhalt von Kanten
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # adaptive Schwelle (gaussian), feine Schrift bleibt erhalten
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    # kleine Punkte wegräumen
    bw = cv2.medianBlur(bw, 3)
    # Als 3‑Kanal zurück (für ReportLab einfacher)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def ensure_portrait(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w > h:  # Querlage -> drehen
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    return img_bgr

# -----------------------------
# PDF-Erzeugung
# -----------------------------
def image_to_pdf_bytes(img_bgr: np.ndarray) -> bytes:
    im_pil = pil_from_cv(img_bgr)
    # Seitenformat an Bild anpassen (72 DPI), auf Portrait erzwingen
    w, h = im_pil.size
    page = portrait((w, h))
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page)
    # Bild randlos platzieren (0,0 unten links)
    c.drawImage(ImageReader(im_pil), 0, 0, width=w, height=h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# -----------------------------
# Upload: tmpfiles.org API (gibt Direkt-Download-Link zurück)
# -----------------------------
async def upload_tmpfiles(pdf_bytes: bytes, filename: str = "scan.pdf") -> str:
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            files = {"file": (filename, pdf_bytes, "application/pdf")}
            r = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
            r.raise_for_status()
            data = r.json()
            url = data.get("data", {}).get("url")
            if not url:
                raise RuntimeError(f"tmpfiles.org Antwort ohne URL: {data}")
            # /api/ gibt Seite; /dl/ Pfad ist Direkt-Download – API liefert bereits /dl/… Link
            # Falls doch nur HTML-Link kam, möglichst /dl/ sicherstellen:
            url = url.replace("https://tmpfiles.org/", "https://tmpfiles.org/")
            return url
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upload fehlgeschlagen: {e}")

# -----------------------------
# Pipeline
# -----------------------------
async def process_url_to_pdf(url: str) -> bytes:
    raw = await fetch_bytes(url)
    pil = pil_from_bytes(raw)
    bgr = cv_from_pil(pil)

    warped = largest_receipt_region(bgr) or bgr
    warped = ensure_portrait(warped)
    scanned = to_scanned_bw(warped)
    pdf_bytes = image_to_pdf_bytes(scanned)
    return pdf_bytes

# -----------------------------
# API Schemas
# -----------------------------
class ScanBody(BaseModel):
    file_url: HttpUrl
    response: str | None = "link"  # "link" oder "pdf"

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "scan-endpoint ok"

@app.get("/scan")
async def scan_get(
    file_url: HttpUrl = Query(..., description="Direkter Bild-URL (Airtable Attachment URL ok)"),
    response: str = Query("link", pattern="^(link|pdf)$"),
):
    pdf_bytes = await process_url_to_pdf(str(file_url))
    if response == "pdf":
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf",
                                 headers={"Content-Disposition": 'inline; filename="scan.pdf"'})
    # sonst Link zurück
    link = await upload_tmpfiles(pdf_bytes, "scan.pdf")
    return JSONResponse({"filename": "scan.pdf", "content_type": "application/pdf", "url": link})

@app.post("/scan")
async def scan_post(payload: ScanBody = Body(...)):
    pdf_bytes = await process_url_to_pdf(str(payload.file_url))
    resp = (payload.response or "link").lower()
    if resp == "pdf":
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf",
                                 headers={"Content-Disposition": 'inline; filename="scan.pdf"'})
    link = await upload_tmpfiles(pdf_bytes, "scan.pdf")
    return JSONResponse({"filename": "scan.pdf", "content_type": "application/pdf", "url": link})
