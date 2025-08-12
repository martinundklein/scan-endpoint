# api/scan.py
import io
import math
import os
import uuid
from typing import Optional, Tuple, List

import httpx
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, Response, PlainTextResponse
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import cv2

app = FastAPI(title="Scan Endpoint", version="2.3.0")

# --------- helpers: io ---------
async def fetch_bytes(url: str, timeout: float = 30.0) -> bytes:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# --------- geometry utils ---------
def order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    return warped

# --------- detection pipeline ---------
def illumination_correct(gray: np.ndarray) -> np.ndarray:
    # Big blur → estimate background → subtract
    bg = cv2.medianBlur(gray, 41)
    corrected = cv2.normalize(cv2.subtract(gray, bg), None, 0, 255, cv2.NORM_MINMAX)
    return corrected

def auto_canny(img: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)

def find_receipt_contour(bgr: np.ndarray) -> Optional[np.ndarray]:
    # downscale for speed
    scale = 900 / max(bgr.shape[:2])
    if scale < 1.0:
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = bgr.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = illumination_correct(gray)

    edges = auto_canny(gray, 0.33)
    # strengthen edges
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = small.shape[:2]
    candidates: List[Tuple[float, np.ndarray]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.02 * W * H:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue
        # narrow receipt aspect (~> 2.2), allow landscape too but prefer tall
        ar = max(w, h) / max(1.0, min(w, h))
        # reject super fat shapes
        if ar < 1.8:
            continue
        # prefer 4-point
        score = area
        if len(approx) == 4:
            score *= 3.0
            pts = approx.reshape(-1, 2)
        else:
            # use box as fallback
            box = cv2.boxPoints(rect)
            pts = box
        # rescale to original coords
        pts = (pts / scale).astype(np.float32)
        candidates.append((score, pts))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def deskew_by_hough(gray: np.ndarray) -> np.ndarray:
    edges = auto_canny(gray, 0.33)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for rho, theta in lines[:, 0]:
            a = (theta * 180 / np.pi) % 180
            # near-vertical lines → compute tilt relative to 90°
            delta = a - 90
            if -45 < delta < 45:
                angles.append(delta)
        if angles:
            angle = np.median(angles)
    if abs(angle) < 0.2:
        return gray
    # rotate around center
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def to_scan_look(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = illumination_correct(gray)
    gray = deskew_by_hough(gray)
    # adaptive threshold to pure b/w
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, 12)
    # small denoise + unsharp
    bw = cv2.medianBlur(bw, 3)
    return bw

def ensure_upright(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h >= w:
        return img
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# --------- pdf & upload ---------
def bw_png_bytes(bw: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bw)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return bytes(buf)

def make_pdf_from_bw(bw: np.ndarray, dpi: int = 300) -> bytes:
    # scale to A4 width keeping aspect
    bw = ensure_upright(bw)
    H, W = bw.shape[:2]
    # convert to PIL for reportlab placement
    img = Image.fromarray(bw)
    # target: A4 with 10mm margin
    a4_w, a4_h = A4  # points (1/72")
    margin = 28.35  # ~10 mm
    max_w = a4_w - 2 * margin
    # compute scale so width fits
    scale = max_w / float(W)
    out_w = max_w
    out_h = float(H) * scale
    if out_h > (a4_h - 2 * margin):
        # scale by height if needed
        out_h = a4_h - 2 * margin
        scale = out_h / float(H)
        out_w = float(W) * scale

    # image bytes for reportlab
    png = io.BytesIO()
    img.save(png, format="PNG")
    png.seek(0)

    mem = io.BytesIO()
    c = canvas.Canvas(mem, pagesize=A4)
    x = (a4_w - out_w) / 2
    y = (a4_h - out_h) / 2
    c.drawImage(png, x, y, width=out_w, height=out_h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    return mem.getvalue()

async def upload_with_fallbacks(data: bytes, filename: str) -> str:
    # 1) tmpfiles.org (simple multipart form)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            files = {"file": (filename, data, "application/pdf")}
            r = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
            if r.status_code == 200:
                j = r.json()
                # tmpfiles returns e.g. {"status":"ok","data":{"url":"https://tmpfiles.org/abcd/file.pdf"}}
                url = j.get("data", {}).get("url")
                if url:
                    return url
    except Exception:
        pass
    # 2) file.io
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            files = {"file": (filename, data, "application/pdf")}
            r = await client.post("https://file.io", files=files)
            if r.status_code == 200 and r.json().get("success"):
                return r.json().get("link")
    except Exception:
        pass
    # 3) 0x0.st (expects raw body; returns short url as text)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post("https://0x0.st", files={"file": (filename, data, "application/pdf")})
            if r.status_code == 200:
                return r.text.strip()
    except Exception:
        pass
    raise HTTPException(502, "Upload fehlgeschlagen (alle Anbieter)")

# --------- main endpoint ---------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "scan-endpoint online. Use /scan?file_url=...&response=link|pdf|json"

@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Öffentlicher Direktlink zum Bild/PDF"),
    response: str = Query("link", pattern="^(link|pdf|json)$"),
):
    # 1) Lade Quelle
    try:
        raw = await fetch_bytes(file_url)
    except Exception:
        raise HTTPException(400, "Konnte Quelle nicht laden (file_url prüfen).")

    # 2) Image öffnen (falls PDF → erste Seite rendern)
    try:
        img = Image.open(io.BytesIO(raw))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception:
        raise HTTPException(415, "Unbekanntes Dateiformat. Bitte Bild/PDF als öffentlich abrufbare URL.")

    # 3) In CV umwandeln
    bgr = pil_to_cv(img)

    # 4) Dokumentkontur suchen & perspektivisch entzerren (oder fallback)
    pts = find_receipt_contour(bgr)
    if pts is not None:
        warped = four_point_transform(bgr, pts)
    else:
        # sanfter crop um Rand/Teppich zu minimieren
        h, w = bgr.shape[:2]
        pad = int(min(h, w) * 0.05)
        warped = bgr[pad:h - pad, pad:w - pad].copy()

    # 5) Scan-Look generieren
    bw = to_scan_look(warped)
    bw = ensure_upright(bw)

    # 6) PDF bauen
    pdf_bytes = make_pdf_from_bw(bw)
    fname = f"scan_{uuid.uuid4().hex[:8]}.pdf"

    # 7) response
    if response == "pdf":
        return Response(pdf_bytes, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{fname}"'})
    elif response == "json":
        # als Base64 (zur Not in Zapier nutzbar)
        import base64
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        return JSONResponse({"filename": fname, "content_type": "application/pdf", "file_base64": b64})

    # default: upload & Link
    link = await upload_with_fallbacks(pdf_bytes, fname)
    return JSONResponse({"url": link, "filename": fname})
