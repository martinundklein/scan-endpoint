# api/scan.py
from fastapi import FastAPI, HTTPException, Query, Response
import httpx, io, math, os
from typing import Tuple, Optional
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv
from skimage.filters import threshold_sauvola
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Receipt Scan Endpoint")

# --------------------------
# Hilfen
# --------------------------
async def fetch_bytes(url: str, timeout: float = 20.0) -> bytes:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Download failed ({r.status_code})")
        return r.content

def to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv.cvtColor(arr, cv.COLOR_BGR2RGB))

def to_bgr(img: Image.Image) -> np.ndarray:
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

def auto_deskew(gray: np.ndarray) -> np.ndarray:
    # Kanten -> Hough-Linien -> Winkelmedian
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, threshold=120)
    if lines is None or len(lines) < 3:
        return gray
    angles = []
    for rho, theta in lines[:,0,:]:
        ang = (theta * 180 / np.pi) - 90  # horizontale Linien -> 0
        # nur fast-horizontale/vertikale Linien
        if -45 <= ang <= 45:
            angles.append(ang)
    if not angles:
        return gray
    angle = float(np.median(angles))
    (h, w) = gray.shape[:2]
    M = cv.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv.warpAffine(gray, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated

def enhance(gray: np.ndarray) -> np.ndarray:
    # milder Rauschfilter + CLAHE für Kontrast
    den = cv.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(den)
    return enh

def try_find_receipt_contour(gray: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    Sucht die größte viereckige Kontur, die KEIN fast-quadratisches Verhältnis hat.
    Liefert (4x2 Punkte) im Uhrzeigersinn + Konfidenz [0..1].
    QR-Codes werden i.d.R. durch quadratisches Verhältnis + kleine Fläche gefiltert.
    """
    h, w = gray.shape[:2]
    area_img = w * h

    # Kanten für Konturen
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blur, 60, 160)

    # leichte Dilatation, damit Kanten schließen
    edges = cv.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0

    for c in cnts:
        area = cv.contourArea(c)
        if area < 0.08*area_img:  # zu klein
            continue

        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02*peri, True)
        if len(approx) != 4:
            continue

        # Bounding-Box Verhältnis prüfen
        rect = cv.minAreaRect(approx)
        (bw, bh) = rect[1]
        if bw == 0 or bh == 0:
            continue
        ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

        # QR-ähnliche Quadrate raus (ratio ~1) ODER sehr kleines Rechteck
        if 0.90 <= ratio <= 1.10:
            continue

        # Score: Fläche*Geradheit
        box = cv.boxPoints(rect)
        box = np.int0(box)

        # "Geradheit": wie gut approx der Kontur folgt
        fill = area / (bw*bh + 1e-6)
        score = (area/area_img) * 0.7 + min(fill, 1.0) * 0.3
        if score > best_score:
            best_score = score
            best = approx.reshape(4,2).astype(np.float32)

    if best is None:
        return None, 0.0

    # Ordnung der Punkte (tl,tr,br,bl)
    def order_pts(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)
    best = order_pts(best)

    # heuristische Konfidenz
    conf = float(min(0.99, best_score))
    return best, conf

def four_point_warp(img: np.ndarray, pts: np.ndarray, out_width: int = 1200) -> np.ndarray:
    (tl, tr, br, bl) = pts
    # Zielgröße nach Seitenlängen
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    if maxW < 10 or maxH < 10:
        return img

    scale = out_width / maxW
    outW = int(maxW * scale)
    outH = int(maxH * scale)

    dst = np.array([[0,0],[outW-1,0],[outW-1,outH-1],[0,outH-1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(pts, dst)
    warped = cv.warpPerspective(img, M, (outW, outH), flags=cv.INTER_CUBIC)
    return warped

def bw_scan(img_rgb: Image.Image, strength: float = 0.6) -> Image.Image:
    """
    Milde B/W-Scan-Optik. strength: 0..1 (0=sehr mild, 1=hart)
    """
    bgr = to_bgr(img_rgb)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray = enhance(gray)
    # Sauvola adaptiv
    win = 25
    k = 0.2 + 0.2*strength  # 0.2..0.4
    thresh = threshold_sauvola(gray, window_size=win, k=k)
    bw = (gray > thresh).astype(np.uint8) * 255

    # etwas Öffnen/Schließen gegen Salz&Pfeffer
    kernel = np.ones((2,2), np.uint8)
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel, iterations=1)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel, iterations=1)

    # leichte Mischung mit Graustufen, damit es nicht „kaputt-hart“ wirkt
    alpha = 0.15 + 0.25*strength  # 0.15..0.40
    mixed = cv.addWeighted(gray, 1-alpha, bw, alpha, 0)
    return to_pil(mixed)

def make_pdf_from_image(img: Image.Image, dpi: int = 300) -> bytes:
    # weißer Rand, dann als PNG in PDF
    img = ImageOps.exif_transpose(img)
    # Für ReportLab
    png_buf = io.BytesIO()
    img.save(png_buf, format="PNG", optimize=True)
    png_buf.seek(0)
    reader = ImageReader(png_buf)

    # Seitenmaß in Punkten
    w_px, h_px = img.size
    w_pt = w_px / dpi * 72
    h_pt = h_px / dpi * 72

    pdf = io.BytesIO()
    c = canvas.Canvas(pdf, pagesize=(w_pt, h_pt))
    c.drawImage(reader, 0, 0, width=w_pt, height=h_pt, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    pdf.seek(0)
    return pdf.read()

# --------------------------
# API
# --------------------------
@app.get("/")
def root():
    return {"ok": True, "usage": "/scan?file_url=...&detect_conf=0.7&bw_strength=0.55"}

@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Öffentliche Bild-URL (Airtable File URL)"),
    detect_conf: float = Query(0.70, ge=0.0, le=1.0, description="Konfidenzschwelle fürs Zuschneiden"),
    bw_strength: float = Query(0.55, ge=0.0, le=1.0, description="Intensität der B/W-Optik 0..1"),
    filename: str = Query("scan.pdf")
):
    try:
        raw = await fetch_bytes(file_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download error: {e}")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        # falls PDF reinkommt -> erste Seite als Bild versuchen
        raise HTTPException(status_code=415, detail="Unsupported file (expecting an image)")

    # Nach EXIF ausrichten & auf sinnvolle Max-Kante skalieren
    img = ImageOps.exif_transpose(img)
    max_side = 2000
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    bgr = to_bgr(img)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    # Deskew + Enhancement
    gray = auto_deskew(gray)
    gray = enhance(gray)

    # Versuch: Beleg finden
    quad, conf = try_find_receipt_contour(gray)

    if quad is not None and conf >= detect_conf:
        warped = four_point_warp(bgr, quad, out_width=1300)
        gray2 = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        gray2 = enhance(gray2)
        result_img = bw_scan(to_pil(gray2), strength=bw_strength)
    else:
        # kein sicherer Cut -> ganze Seite milde B/W
        result_img = bw_scan(to_pil(gray), strength=bw_strength)

    pdf_bytes = make_pdf_from_image(result_img, dpi=300)
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "application/pdf",
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
