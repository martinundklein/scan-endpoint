from fastapi import FastAPI, Query, Response, HTTPException
import httpx, io, math, os
import numpy as np
import cv2 as cv
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Scan Endpoint", version="1.0.0")

# -------- Helpers --------

async def fetch_image(url: str, timeout_s: float = 25.0) -> np.ndarray:
    headers = {"User-Agent": "scan-endpoint/1.0"}
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    async with httpx.AsyncClient(headers=headers, timeout=timeout_s, limits=limits, follow_redirects=True) as client:
        last_err = None
        for _ in range(3):
            try:
                r = await client.get(url)
                r.raise_for_status()
                data = np.frombuffer(r.content, np.uint8)
                img = cv.imdecode(data, cv.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Konnte Bild nicht dekodieren")
                return img
            except Exception as e:
                last_err = e
        raise HTTPException(status_code=400, detail=f"Download/Decode fehlgeschlagen: {last_err}")

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if len(img_bgr.shape) == 2:
        return img_bgr
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

def deskew_small_angles(gray: np.ndarray) -> np.ndarray:
    # Nur leichte Schiefstellung korrigieren (max ~5°)
    edges = cv.Canny(gray, 50, 150)
    lines = cv.HoughLines(edges, 1, np.pi/180, threshold=120)
    if lines is None:
        return gray
    angles = []
    for rho, theta in lines[:,0]:
        a = theta
        # nur nahe 0°/180° oder 90° berücksichtigen
        deg = (a * 180.0 / math.pi) % 180.0
        if deg < 10 or abs(deg-90) < 10 or abs(deg-180) < 10:
            # Winkel normalisieren auf -90..90
            ang = deg if deg <= 90 else deg-180
            angles.append(ang)
    if not angles:
        return gray
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray
    if abs(median_angle) > 5:
        median_angle = 5.0 if median_angle > 0 else -5.0
    (h, w) = gray.shape[:2]
    M = cv.getRotationMatrix2D((w/2, h/2), median_angle, 1.0)
    rotated = cv.warpAffine(gray, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return rotated

def detect_document_polygon(gray: np.ndarray):
    # Kanten -> Konturen -> bestes 4‑Eck
    h, w = gray.shape
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blur, 50, 150)
    # QR-Code stört oft: Kleine Quadrate filtern wir später raus
    cnts, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_area = float(h*w)
    best = None
    best_score = 0.0
    for c in cnts:
        area = cv.contourArea(c)
        if area < img_area * 0.05:  # ignoriere sehr kleine Konturen
            continue
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        rect = cv.minAreaRect(approx)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0:
            continue
        aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
        # Filtere nahezu quadratische, kleine Flächen (typisch QR):
        if 0.85 <= aspect <= 1.15 and area < img_area * 0.25:
            continue
        # Score: Flächenanteil * rechteckige Qualität
        area_ratio = area / img_area
        score = area_ratio
        if score > best_score:
            best_score = score
            best = approx
    return best, float(best_score)

def order_corners(pts: np.ndarray):
    pts = pts.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(4)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, bl, br], dtype=np.float32)

def warp_to_topdown(src_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    h, w = src_bgr.shape[:2]
    pts = order_corners(quad)
    (tl, tr, bl, br) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    maxW = max(100, maxW)
    maxH = max(100, maxH)
    dst = np.array([[0,0],[maxW-1,0],[0,maxH-1],[maxW-1,maxH-1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(pts, dst)
    return cv.warpPerspective(src_bgr, M, (maxW, maxH), flags=cv.INTER_CUBIC)

def bw_scan(gray: np.ndarray, strength: float = 0.85) -> np.ndarray:
    # CLAHE für Kontrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    # adaptive Threshold (Blockgröße abhängig von Bildbreite)
    h, w = g.shape
    block = int(max(31, (min(h, w) // 30) | 1))  # ungerade
    th = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, block, 7)
    # sanfte Mischung mit Graustufe, um "zu hart" zu vermeiden
    mixed = cv.addWeighted(th, strength, g, 1.0 - strength, 0)
    return mixed

def upscale_if_needed(img: np.ndarray, mode: str = "auto") -> np.ndarray:
    if mode not in ("auto","off"):
        mode = "auto"
    if mode == "off":
        return img
    h, w = img.shape[:2]
    long = max(h, w)
    if long >= 1600:
        return img
    scale = 1600.0 / long
    new = (int(w*scale), int(h*scale))
    return cv.resize(img, new, interpolation=cv.INTER_CUBIC)

def make_pdf_from_image(pil_img: Image.Image, max_kb: int, hard_cap_kb: int) -> bytes:
    # 1) zu Graustufe, moderate Zielgröße
    pil = pil_img.convert("L")
    max_long = 1600
    w, h = pil.size
    long_side = max(w, h)
    if long_side > max_long:
        scale = max_long / float(long_side)
        pil = pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    # 2) JPEG-Qualitätsschleife + ggf. Downscale
    quality = 75
    data = None
    for _ in range(8):
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        jpg = buf.getvalue()
        if len(jpg) <= max_kb * 1024:
            data = jpg
            break
        # sonst: Qualität runter, dann ggf. kippe noch die Größe
        if quality > 45:
            quality -= 8
        else:
            # verkleinere 10%
            w, h = pil.size
            pil = pil.resize((int(w*0.9), int(h*0.9)), Image.LANCZOS)
    if data is None:
        data = jpg
        if len(data) > hard_cap_kb * 1024:
            # letzter Rettungsanker: noch kleiner
            w, h = pil.size
            pil = pil.resize((max(600, int(w*0.85)), max(600, int(h*0.85))), Image.LANCZOS)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=42, optimize=True, progressive=True)
            data = buf.getvalue()

    # 3) in PDF einbetten (Seite auf Bildgröße, zentriert auf A4)
    img_reader = ImageReader(io.BytesIO(data))
    a4_w, a4_h = A4
    # Bildgröße in Punkten (bei 72dpi: 1px ~ 0.75pt). Wir nehmen dynamisch: skaliere so, dass es breit passt.
    with Image.open(io.BytesIO(data)) as tmp:
        iw, ih = tmp.size
    # target Breite = 500pt – 540pt (optisch), aber wenn hochkant sehr lang: auf A4 Breite
    max_pt_w = a4_w - 60
    scale = min(max_pt_w / iw, (a4_h - 60) / ih)
    pw, ph = iw * scale, ih * scale
    # PDF bauen
    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    x = (a4_w - pw) / 2
    y = (a4_h - ph) / 2
    c.drawImage(img_reader, x, y, pw, ph, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    return out.getvalue()

def np_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv.cvtColor(arr, cv.COLOR_BGR2RGB))

# -------- Routes --------

@app.get("/")
async def root():
    return {"ok": True, "service": "scan-endpoint", "version": "1.0.0"}

@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Direkte Bild-URL (JPG/PNG…)"),
    filename: str = Query("scan.pdf"),
    crop_conf: float = Query(0.70, ge=0.0, le=1.0),
    bw_strength: float = Query(0.85, ge=0.4, le=1.0),
    max_kb: int = Query(600, ge=60, le=4000),
    hard_cap_kb: int = Query(900, ge=80, le=8000),
    upscale: str = Query("auto", description="'auto' oder 'off'"),
):
    # 1) Download
    bgr = await fetch_image(file_url)
    # 2) ggf. Upscale
    bgr = upscale_if_needed(bgr, mode=upscale)
    # 3) Grund-Graustufe + leichte Deskew
    gray = to_gray(bgr)
    gray = deskew_small_angles(gray)
    # 4) Dokumentenerkennung (nur wenn sicher)
    quad, score = detect_document_polygon(gray)
    if quad is not None and score >= crop_conf:
        try:
            bgr = warp_to_topdown(cv.cvtColor(gray, cv.COLOR_GRAY2BGR), quad)
            gray = to_gray(bgr)
        except Exception:
            # Falls Warp scheitert: ignorieren
            pass
    # 5) B/W-Scan (schonend)
    bw = bw_scan(gray, strength=bw_strength)
    pil = np_to_pil(bw)
    # 6) PDF erzeugen + Komprimieren
    pdf_bytes = make_pdf_from_image(pil, max_kb=max_kb, hard_cap_kb=hard_cap_kb)

    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "Cache-Control": "no-store",
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
