# api/scan.py
import io
import math
from typing import Optional, Tuple

import numpy as np
import cv2
import httpx
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import Response, PlainTextResponse
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Receipt Scan Endpoint")

# ---------- helpers ----------

def to_np(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def resize_for_process(bgr: np.ndarray, max_side: int = 2000) -> Tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return bgr, scale

def detect_qr_like_masks(gray: np.ndarray) -> np.ndarray:
    """Erkennt große fast-quadratische Blöcke (QR/Datamatrix) und maskiert sie weg."""
    h, w = gray.shape
    mask = np.zeros_like(gray, dtype=np.uint8)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (h*w)*0.01:  # zu klein -> ignorieren
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ratio = bw / float(bh + 1e-6)
        fill = area / float(bw*bh + 1e-6)
        # quadratisch + dicht gefüllt => sehr wahrscheinlich QR-Code
        if 0.8 <= ratio <= 1.25 and fill > 0.6:
            cv2.drawContours(mask, [c], -1, 255, -1)
    return mask

def doc_candidate(gray: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """liefert beste 4‑Punkt‑Kontur und eine rohe Güte."""
    h, w = gray.shape
    # leicht glätten
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Kanten
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    # QR‑ähnliche Bereiche entfernen
    qr_mask = detect_qr_like_masks(gray)
    edges = cv2.bitwise_and(edges, cv2.bitwise_not(qr_mask))

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    img_area = float(h*w)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * 0.05:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Rechteck‑Güte via 4 Ecken
        if len(approx) == 4:
            pts = approx.reshape(-1, 2)
        else:
            pts = box.reshape(-1, 2)

        # Ordnung: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        ordered = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)]
        ], dtype=np.float32)

        # Metrics
        rect_area = cv2.contourArea(ordered)
        if rect_area <= 0:
            continue
        solidity = area / (rect_area + 1e-6)
        fill = area / img_area
        # Kontrast in dieser Region
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, ordered.astype(np.int32), 255)
        roi = cv2.bitwise_and(gray, gray, mask=mask)
        cstd = float(np.std(roi[mask==255]))  # grober Kontrast

        score = (min(fill/0.7,1.0) * 0.45) + (min(solidity,1.0) * 0.35) + (min(cstd/30.0,1.0) * 0.20)

        if score > best_score:
            best_score = score
            best = ordered

    return (best if best is not None else None, float(best_score))

def four_point_warp(bgr: np.ndarray, pts: np.ndarray, pad: int = 12) -> np.ndarray:
    (tl, tr, br, bl) = pts.astype(np.float32)
    def dist(a,b): return np.linalg.norm(a-b)
    wA = dist(br, bl)
    wB = dist(tr, tl)
    hA = dist(tr, br)
    hB = dist(tl, bl)
    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(bgr, M, (maxW, maxH), flags=cv2.INTER_CUBIC)
    # leichter Rand
    warped = cv2.copyMakeBorder(warped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255,255,255))
    return warped

def enhance_bw(bgr: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """CLAHE + adaptives B/W + sanftes Schärfen. Stärke 0..1"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clip = 2.0 + 2.0*strength
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    g = clahe.apply(gray)
    # adaptives Threshold
    block = int(21 + strength*20)  # 21..41
    if block % 2 == 0:
        block += 1
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block, 10)
    # milde Entzerrung Helligkeit
    bw = cv2.medianBlur(bw, 3)
    # leicht schärfen
    if strength > 0:
        k = 0.5 + 0.8*strength  # 0.5..1.3
        blurred = cv2.GaussianBlur(bw, (0,0), 1.0)
        sharp = cv2.addWeighted(bw, 1+k, blurred, -k, 0)
        bw = sharp
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def portrait_rotate(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    return bgr if h >= w else cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

def make_pdf_bytes(img: Image.Image) -> bytes:
    # Seite = A4 hoch
    page_w, page_h = A4
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    # Bild als temporärer ImageReader aus Bytes
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    ir = ImageReader(bio)
    iw, ih = img.size
    # proportional einpassen mit Rand
    margin = 24
    max_w = page_w - 2*margin
    max_h = page_h - 2*margin
    scale = min(max_w/iw, max_h/ih)
    ow, oh = iw*scale, ih*scale
    x = (page_w - ow)/2
    y = (page_h - oh)/2
    c.drawImage(ir, x, y, ow, oh, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    return buf.getvalue()

# ---------- endpoint ----------

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK – use /scan?file_url=...&crop_conf=0.70&enhance=0.55"

@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Öffentliche Bild-URL (Airtable CDN etc.)"),
    crop_conf: float = Query(0.70, ge=0.0, le=1.0, description="Schwellwert für Cropping/Perspektive"),
    enhance: float = Query(0.55, ge=0.0, le=1.0, description="Stärke der B/W‑Aufbereitung (0..1)"),
) -> Response:
    # Bild laden
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(file_url, follow_redirects=True)
    if r.status_code != 200:
        raise HTTPException(400, f"Bild-Download fehlgeschlagen ({r.status_code})")
    try:
        img = Image.open(io.BytesIO(r.content))
    except Exception as e:
        raise HTTPException(400, f"Ungültiges Bild: {e}")

    bgr = to_np(img)
    bgr = portrait_rotate(bgr)
    bgr_small, scale = resize_for_process(bgr, 1800)

    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    pts, score = doc_candidate(gray)

    if pts is not None and score >= crop_conf:
        warped = four_point_warp(bgr_small, pts)
        # zurück auf „Original“‑Größe hochrechnen, aber capped
        warped = resize_for_process(warped, 2200)[0]
        work = enhance_bw(warped, strength=enhance)
    else:
        # Fallback: kein aggressives Cropping
        work = enhance_bw(bgr_small, strength=max(0.35, enhance*0.8))

    # PDF erzeugen
    pdf_img = to_pil(work)
    pdf_bytes = make_pdf_bytes(pdf_img)

    headers = {
        "Content-Type": "application/pdf",
        "Content-Disposition": 'inline; filename="scan.pdf"',
        "X-Scan-Decision": ("crop" if (pts is not None and score >= crop_conf) else "no-crop"),
        "X-Scan-Score": f"{score:.3f}",
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
