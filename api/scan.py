import io
import math
from typing import Tuple, Optional, List

import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
import httpx
from PIL import Image
import cv2 as cv

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Simple Receipt Scanner")


# ---------- Helpers ----------

async def fetch_bytes(url: str, timeout: float = 30.0) -> bytes:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout,
                                     headers={"User-Agent": "scan-endpoint/1.0"}) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Download fehlgeschlagen: {e}") from e


def load_bgr(img_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv.imdecode(data, cv.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=415, detail="Bildformat wird nicht unterstützt.")
    return bgr


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    # sanft glätten, dann adaptive threshold (gaussian)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    bw = cv.adaptiveThreshold(
        blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 10
    )
    return bw


def estimate_skew_angle(gray: np.ndarray) -> float:
    """kleine Schätzung des Schiefwinkels in Grad (horizontaler Text)"""
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180.0, threshold=max(120, int(0.15 * gray.shape[1])))
    if lines is None:
        return 0.0
    # Winkel um 0° sammeln (±15°)
    angles = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        angle = (theta * 180.0 / np.pi) - 90.0  # 0≈horizontal
        if -15.0 <= angle <= 15.0:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_keep_size(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.3:
        return image
    h, w = image.shape[:2]
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """ordnet vier Ecken: [tl, tr, br, bl]"""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = order_corners(pts)
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dest = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dest)
    return cv.warpPerspective(image, M, (maxW, maxH), flags=cv.INTER_CUBIC)


def contour_confidence(cnt: np.ndarray, img_area: float) -> Tuple[float, dict]:
    area = cv.contourArea(cnt)
    if area <= 0:
        return 0.0, {}
    x, y, w, h = cv.boundingRect(cnt)
    aspect = h / max(w, 1)
    area_ratio = area / img_area
    rectangularity = area / (w * h + 1e-6)

    score = 0.0
    # groß genug?
    if area_ratio > 0.20: score += 0.25
    if area_ratio > 0.35: score += 0.20  # sehr gut
    # länglich (Beleg) – QR-Codes (≈1:1) fallen raus
    if aspect > 1.3: score += 0.25
    if aspect > 1.8: score += 0.10
    # rechteck-nah
    if rectangularity > 0.75: score += 0.15
    if rectangularity > 0.85: score += 0.05

    # vier Ecken?
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4: score += 0.10

    details = dict(area_ratio=float(area_ratio), aspect=float(aspect),
                   rectangularity=float(rectangularity), approx_pts=int(len(approx)))
    return score, details


def detect_receipt(gray: np.ndarray) -> Tuple[Optional[np.ndarray], float, dict]:
    """liefert ggf. 4-Punkte-Polygon (float32) + Score"""
    img_area = float(gray.shape[0] * gray.shape[1])

    # Kanten und Konturen
    bw = adaptive_binarize(gray)
    if np.mean(bw) < 127:  # invertiert? wir wollen Text schwarz
        bw = 255 - bw
    edges = cv.Canny(bw, 60, 180)
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best_pts = None
    best_score = 0.0
    best_meta = {}
    for cnt in contours:
        score, meta = contour_confidence(cnt, img_area)
        if score > best_score:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) >= 4:
                # nehme die 4 “besten” Ecken via minAreaRect/box
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect).astype(np.float32)
                best_pts = box
                best_score = score
                best_meta = meta

    return best_pts, float(best_score), best_meta


def to_pdf_bytes(pil_img: Image.Image) -> bytes:
    # PNG in Ram
    buf_png = io.BytesIO()
    pil_img.save(buf_png, format="PNG", optimize=True)
    buf_png.seek(0)

    # PDF mit gleicher Größe in ~200 DPI (gute Lesbarkeit, kleine Datei)
    dpi = 200
    w_px, h_px = pil_img.size
    w_in = w_px / dpi
    h_in = h_px / dpi

    buf_pdf = io.BytesIO()
    c = canvas.Canvas(buf_pdf, pagesize=(w_in * inch, h_in * inch))
    c.drawImage(ImageReader(buf_png), 0, 0, width=w_in * inch, height=h_in * inch, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    buf_pdf.seek(0)
    return buf_pdf.read()


# ---------- Pipeline ----------

def process_image(bgr: np.ndarray, min_crop_conf: float = 0.7, bw_strength: float = 0.6) -> Image.Image:
    """
    - nur croppen/geradeziehen, wenn Sicherheit >= min_crop_conf
    - sonst nur leicht deskew + B/W
    """
    # verkleinern für schnellere Konturerkennung (nicht fürs Endresultat!)
    h, w = bgr.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1400 else 1.0
    small = cv.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    small_gray = to_gray(small)

    # Kandidat finden
    pts, score, _meta = detect_receipt(small_gray)

    # in Originalkoordinaten mappen
    warped = None
    if pts is not None and score >= min_crop_conf:
        pts_orig = (pts / scale).astype(np.float32)
        warped = four_point_warp(bgr, pts_orig)

    work = warped if warped is not None else bgr

    # leichte Schiefstandkorrektur
    gray = to_gray(work)
    angle = estimate_skew_angle(gray)
    work = rotate_keep_size(work, angle)

    # zarte B/W-Optik
    gray = to_gray(work)
    bw = adaptive_binarize(gray)

    # “Stärke” mischen: bw_strength in [0..1] mischt mit leichtem Kontrastbild
    #  -> weicher, besser lesbar als hartes Schwarz/Weiß
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    mixed = cv.addWeighted(eq, (1.0 - bw_strength), bw, bw_strength, 0)

    # zurück zu PIL (RGB)
    pil = Image.fromarray(mixed).convert("L")  # Graustufen-PDF ist kleiner
    return pil


# ---------- Routes ----------

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK - /scan?file_url=ENCODED_URL  (optional: min_conf, bw)"


@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Öffentliche Bild-URL (z.B. Airtable v5 URL)"),
    min_conf: float = Query(0.70, ge=0.0, le=1.0, description="Mindest-Sicherheit fürs Zuschneiden"),
    bw: float = Query(0.60, ge=0.0, le=1.0, description="B/W Intensität (0=weich, 1=hart)")
):
    """
    Liefert direkt ein PDF (Content-Disposition: attachment; filename=scan.pdf)
    """
    img_bytes = await fetch_bytes(file_url)
    bgr = load_bgr(img_bytes)

    pil = process_image(bgr, min_crop_conf=min_conf, bw_strength=bw)
    pdf_bytes = to_pdf_bytes(pil)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="scan.pdf"'}
    )
