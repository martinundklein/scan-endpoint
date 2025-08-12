import io
import math
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from PIL import Image, ImageOps
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Receipt Scan API (safe mode)")

# ---------- Helpers ----------

async def fetch_image_bytes(url: str) -> bytes:
    timeout = httpx.Timeout(30.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, follow_redirects=True)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Download failed: {r.status_code}")
        return r.content

def pil_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b))
    # Immer nach RGB konvertieren, einige JPEGs kommen als “L”/“P”
    return img.convert("RGB")

def to_cv(img: Image.Image) -> np.ndarray:
    # PIL RGB -> OpenCV BGR
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def to_pil(arr_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def deskew_soft(gray: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """
    Sanftes Begradigen: schätzt kleinen Schräglauf in [-max_angle, max_angle] Grad.
    Schneidet NICHT, füllt Ränder weiß.
    """
    # Leichte Kantenverstärkung
    edges = cv2.Canny(gray, 60, 180, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)

    if lines is None or len(lines) == 0:
        return gray

    # Winkel sammeln (in Grad), Linien nahe 0° bzw. 90° berücksichtigen
    angles = []
    for rho_theta in lines:
        rho, theta = rho_theta[0]
        # Normiere auf [-90, 90)
        angle = (theta * 180.0 / np.pi) - 90.0
        # Auf Nähe zu 0° oder 90° bringen
        if angle < -45:
            angle += 90
        if angle > 45:
            angle -= 90
        angles.append(angle)

    if not angles:
        return gray

    # Robust: Median statt Mittelwert
    med = float(np.median(angles))
    if abs(med) > max_angle:
        # Sicherheit: nicht stärker als max_angle drehen
        med = max(-max_angle, min(max_angle, med))

    if abs(med) < 0.2:
        return gray  # praktisch gerade

    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), med, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255  # Weiß auffüllen
    )
    return rotated

def bw_scan_safe(img: Image.Image, strength: float = 0.6) -> Image.Image:
    """
    'Scan'-Look ohne Zuschneiden/Perspektive.
    Schritte:
      - sanftes Denoise
      - CLAHE Kontrast
      - adaptives Thresholding (schonend)
    strength in [0..1]: 0 = weich, 1 = knackig
    """
    bgr = to_cv(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Sanftes Denoising (Detail erhalten)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # Sanftes Begradigen, begrenze Winkel
    gray = deskew_soft(gray, max_angle=4.0)

    # Kontrast (CLAHE)
    # Clip-Limit je nach Stärke
    clip = 2.0 + 2.0 * float(strength)  # 2.0..4.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Adaptives Thresholding (Gaussian) – blockSize odd, C fein abstimmen
    block = int(21 + 10 * strength)  # 21..31
    if block % 2 == 0:
        block += 1
    C = int(8 + 6 * strength)        # 8..14

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block,
        C=C
    )

    # leichte Öffnung, um “Salz” zu reduzieren – sehr mild
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    # Zurück nach PIL (als 1-Kanal Bild), dann zu RGB/“weiß” Hintergrund
    pil_bw = Image.fromarray(bw, mode="L")
    # Ein bisschen weicher: leichtes Minimum an Anti-Alias für PDF
    pil_bw = ImageOps.autocontrast(pil_bw, cutoff=0)
    return pil_bw.convert("RGB")

def pdf_from_pil(img: Image.Image) -> bytes:
    """
    Legt das Bild ohne Zuschneiden auf eine A4-Seite, skaliert mit Erhalt des Seitenverhältnisses.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4

    # PIL -> bytes für ReportLab
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)
    rl_img = ImageReader(img_buf)

    # Zielgröße berechnen (mit Rändern)
    margin = 20  # Punkte
    max_w = page_w - 2 * margin
    max_h = page_h - 2 * margin

    iw, ih = img.size
    scale = min(max_w / iw, max_h / ih)
    out_w = iw * scale
    out_h = ih * scale

    x = (page_w - out_w) / 2
    y = (page_h - out_h) / 2

    c.drawImage(rl_img, x, y, width=out_w, height=out_h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    return buf.getvalue()

# ---------- Routes ----------

@app.get("/", summary="Health")
async def root():
    return {"ok": True, "service": "receipt-scan-safe", "version": "1.0.0"}

@app.get(
    "/scan",
    summary="Erzeuge Scan-PDF (safe)",
    response_class=Response
)
async def scan(
    file_url: str = Query(..., description="Direkt-URL zum Bild (JPG/PNG)"),
    strength: float = Query(0.6, ge=0.0, le=1.0, description="Scan-Intensität 0..1 (Standard 0.6)")
):
    """
    Holt ein Bild per URL, wendet sanften Scan-Look an (ohne Cropping/Perspektive),
    und liefert ein A4-PDF (application/pdf) zurück.
    """
    try:
        raw = await fetch_image_bytes(file_url)
        pil = pil_from_bytes(raw)
        scanned = bw_scan_safe(pil, strength=strength)
        pdf_bytes = pdf_from_pil(scanned)
        # Direkt als Datei ausliefern, damit Zapier den Binary-Body bekommt
        headers = {
            "Content-Disposition": 'inline; filename="scan.pdf"'
        }
        return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"processing error: {e}")
