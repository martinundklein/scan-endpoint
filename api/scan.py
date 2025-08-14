import io
import math
from typing import Optional, Tuple

import numpy as np
import httpx
from PIL import Image, ImageFilter, ImageOps
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import StreamingResponse, PlainTextResponse
import cv2 as cv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, portrait
from reportlab.lib.utils import ImageReader

# ------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------
app = FastAPI(title="Lightweight Scan API", version="1.1.0")


@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"


@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "healthy"


# ------------------------------------------------------------
# Helper: download image
# ------------------------------------------------------------
async def fetch_bytes(url: str, timeout: float = 25.0) -> bytes:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, follow_redirects=True)
        if r.status_code != 200:
            raise HTTPException(400, f"Download failed: HTTP {r.status_code}")
        data = r.content
        if not data:
            raise HTTPException(400, "Empty file")
        return data


# ------------------------------------------------------------
# Helper: cv <-> PIL
# ------------------------------------------------------------
def imread_from_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        # evtl. ist es schon ein PDF -> ablehnen, denn wir erwarten Rasterbild
        raise HTTPException(415, "Unsupported file type (expecting JPEG/PNG)")
    return img


def to_pil(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def to_bgr(img_pil: Image.Image) -> np.ndarray:
    return cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)


# ------------------------------------------------------------
# Qualitätsmetriken
# ------------------------------------------------------------
def metrics(gray: np.ndarray) -> Tuple[float, float, int]:
    """returns (sharpness_var, contrast_std, min_dim)"""
    sharp = cv.Laplacian(gray, cv.CV_64F).var()
    contrast = float(gray.std())
    h, w = gray.shape[:2]
    return sharp, contrast, min(h, w)


# ------------------------------------------------------------
# vorsichtiger Deskew (nur kleine Winkel, robust)
# ------------------------------------------------------------
def estimate_skew_angle(gray: np.ndarray) -> float:
    # Kanten + Hough-Linien
    edges = cv.Canny(gray, 60, 180)
    lines = cv.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        # vertikale Linien aussortieren, sonst kippt das Ergebnis
        angle = (theta * 180.0 / math.pi) - 90.0
        if -45 < angle < 45:
            angles.append(angle)
    if not angles:
        return 0.0
    # Median ist robuster
    ang = float(np.median(angles))
    # Begrenzen, um Artefakte zu vermeiden
    return float(np.clip(ang, -10.0, 10.0))


def rotate_bound(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.1:
        return bgr
    h, w = bgr.shape[:2]
    center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv.warpAffine(bgr, M, (nW, nH), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)


# ------------------------------------------------------------
# vorsichtiger Crop (nur wenn eindeutig)
# - sucht größtes „quasi‑rechteckiges“ helles Blatt
# - ignoriert kleine, fast quadratische (QR) Konturen
# ------------------------------------------------------------
def safe_crop(bgr: np.ndarray, conf: float = 0.7) -> np.ndarray:
    h, w = bgr.shape[:2]
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    # leichte Glättung hilft beim Konturfinden
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # adaptives Threshold sorgt für weißen Beleg auf dunklem Boden
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, 35, 5)
    th = cv.medianBlur(th, 3)
    # Invertieren: Papier als große weiße Fläche
    binv = 255 - th

    cnts, _ = cv.findContours(binv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr

    # größter Kandidat
    cnt = max(cnts, key=cv.contourArea)
    area = cv.contourArea(cnt)
    area_ratio = area / float(h * w)

    # zu klein? -> abbrechen
    if area_ratio < 0.35:  # 35% der Fläche
        return bgr

    # polygonisieren und Rechteck testen
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        # Seitenverhältnis prüfen (Belege meist hochkant)
        rect = cv.minAreaRect(cnt)
        (cx, cy), (rw, rh), _ = rect
        if rw == 0 or rh == 0:
            return bgr
        ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
        # QR-Codes sind fast quadratisch (<1.3). Wir wollen >1.5 für Belege
        if ratio < 1.5:
            return bgr

        # „Confidence“ über Rect-Füllgrad
        box = cv.boxPoints(rect)
        box = np.int0(box)
        rect_area = float(rw * rh)
        fill = area / (rect_area + 1e-6)  # 1.0 = perfekt
        if fill < conf:  # zu „zitterig“ erkannt
            return bgr

        # Perspektiv-Transform auf axis-aligned
        # Sortiere Punkte: tl, tr, br, bl
        pts = box.astype("float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        src = np.array([tl, tr, br, bl], dtype=np.float32)

        # Zielbreite/‑höhe
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxW = int(max(widthA, widthB))
        maxH = int(max(heightA, heightB))
        if maxW < 200 or maxH < 200:  # zu klein, lieber lassen
            return bgr
        dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
        M = cv.getPerspectiveTransform(src, dst)
        warped = cv.warpPerspective(bgr, M, (maxW, maxH), flags=cv.INTER_CUBIC)
        return warped

    # kein sauberes Rechteck -> nicht croppen
    return bgr


# ------------------------------------------------------------
# Scan-Looks (sanft / normal / hart) – alle ohne cv.ximgproc
# ------------------------------------------------------------
def scan_gentle(bgr: np.ndarray) -> np.ndarray:
    """nur leicht aufhellen/schärfen, keine harte Binärisierung"""
    pil = to_pil(bgr).convert("L")
    pil = ImageOps.autocontrast(pil, cutoff=1)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=6))
    return to_bgr(pil.convert("RGB"))


def scan_soft(bgr: np.ndarray) -> np.ndarray:
    """grau + etwas Kontrast, leichte Schwelle (Otsu gemischt)"""
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    # CLAHE für lokale Kontraste
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)
    # Otsu
    _, otsu = cv.threshold(g2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Mischung: 65% binär, 35% grau -> „Scan“-Look, aber nicht brutal
    mix = cv.addWeighted(otsu, 0.65, g2, 0.35, 0)
    return cv.cvtColor(mix, cv.COLOR_GRAY2BGR)


def scan_full(bgr: np.ndarray) -> np.ndarray:
    """harter SW-Scan mit adaptivem Threshold, gut lesbar bei gutem Foto"""
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    # Entrauschen, leicht schärfen
    gray = cv.bilateralFilter(gray, 5, 35, 35)
    # Adaptiv (Gaussian), block 31, C=8
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, 31, 8)
    # kleine Flecken entfernen
    bw = cv.medianBlur(bw, 3)
    return cv.cvtColor(bw, cv.COLOR_GRAY2BGR)


# ------------------------------------------------------------
# PDF-Erstellung mit Qualitätsbudget
# ------------------------------------------------------------
def make_pdf(image_bgr: np.ndarray, filename: str, max_kb: int, hard_cap_kb: int) -> bytes:
    # Fit auf A4-Hochformat (mit 12 mm Rand)
    a4_w, a4_h = portrait(A4)
    margin = 34  # ~12 mm @72dpi
    avail_w = a4_w - 2 * margin
    avail_h = a4_h - 2 * margin

    pil = to_pil(image_bgr).convert("L")  # Graustufen-PDF spart Platz

    # Qualitätsschleife (JPEG in PDF) – starte moderat
    for quality in [85, 75, 65, 55, 50, 45, 40]:
        # Bildgröße auf A4 nutzbar skalieren
        iw, ih = pil.size
        scale = min(avail_w / iw, avail_h / ih, 1.0)
        new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        pil_resized = pil if scale == 1.0 else pil.resize(new_size, Image.LANCZOS)

        # als JPEG in Memory
        img_buf = io.BytesIO()
        pil_resized.save(img_buf, format="JPEG", quality=quality, optimize=True)
        img_buf.seek(0)

        # PDF rendern
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=portrait(A4))
        img_reader = ImageReader(img_buf)  # wichtig: ImageReader statt BytesIO direkt
        # zentriert platzieren
        draw_w, draw_h = pil_resized.size
        x = (a4_w - draw_w) / 2
        y = (a4_h - draw_h) / 2
        c.drawImage(img_reader, x, y, width=draw_w, height=draw_h)
        c.showPage()
        c.save()
        data = pdf_buf.getvalue()
        kb = len(data) // 1024
        if kb <= max_kb or quality == 40:
            # ok oder wir sind am Ende
            # Sicherheit: nicht über Hard Cap
            if kb > hard_cap_kb:
                # letzte Notbremse: zusätzliche Verkleinerung
                shrink = pil_resized.resize((int(draw_w * 0.92), int(draw_h * 0.92)), Image.LANCZOS)
                img_buf2 = io.BytesIO()
                shrink.save(img_buf2, format="JPEG", quality=40, optimize=True)
                img_buf2.seek(0)
                pdf_buf2 = io.BytesIO()
                c2 = canvas.Canvas(pdf_buf2, pagesize=portrait(A4))
                c2.drawImage(ImageReader(img_buf2),
                             (a4_w - shrink.size[0]) / 2,
                             (a4_h - shrink.size[1]) / 2,
                             width=shrink.size[0], height=shrink.size[1])
                c2.showPage()
                c2.save()
                data = pdf_buf2.getvalue()
            return data

    return data  # Fallback


# ------------------------------------------------------------
# Haupt-Endpoint
# ------------------------------------------------------------
@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Direkter Bild-URL (JPEG/PNG)"),
    filename: str = Query("scan.pdf"),
    max_kb: int = Query(600, ge=80, le=2000),
    hard_cap_kb: int = Query(900, ge=120, le=4000),
    crop_conf: float = Query(0.70, ge=0.50, le=0.95),
    bw_strength: float = Query(0.80, ge=0.0, le=1.0),  # steuert nur Mix im 'soft' Fall
):
    """
    Liefert ein PDF im Scan-Look zurück.
    - Qualitäts-Trigger entscheidet zwischen 'gentle' / 'soft' / 'full'
    - Deskew klein, Crop nur wenn sehr sicher (crop_conf)
    - PDF wird auf Zielgröße heruntergeregelt
    """

    raw = await fetch_bytes(file_url)
    bgr = imread_from_bytes(raw)

    # ggf. Orientation korrigieren (EXIF)
    try:
        pil_in = Image.open(io.BytesIO(raw))
        pil_in = ImageOps.exif_transpose(pil_in)
        bgr = to_bgr(pil_in.convert("RGB"))
    except Exception:
        pass  # falls ohne EXIF, egal

    # leichte Normalisierung (max Breite/Höhe 3000px, um Render zu schonen)
    H, W = bgr.shape[:2]
    max_side = 3000
    if max(H, W) > max_side:
        scale = max_side / float(max(H, W))
        bgr = cv.resize(bgr, (int(W * scale), int(H * scale)), interpolation=cv.INTER_AREA)

    gray0 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    # Qualitätsmessung
    sharp, contrast, mindim = metrics(gray0)

    # Entscheidungslogik
    # (Grenzen bewusst konservativ gewählt; anpassbar)
    if sharp > 220 and contrast > 42 and mindim >= 1100:
        mode = "full"
    elif sharp > 120 and contrast > 30 and mindim >= 900:
        mode = "soft"
    else:
        mode = "gentle"

    # evtl. leicht hochskalieren wenn sehr klein/unscharf
    if mode == "gentle" and mindim < 1000:
        scale = 1000.0 / (mindim + 1e-6)
        scale = float(np.clip(scale, 1.0, 1.6))  # max 1.6x
        bgr = cv.resize(bgr, (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale)),
                        interpolation=cv.INTER_CUBIC)

    # deskew nur wenn kleiner Winkel erkannt
    ang = estimate_skew_angle(cv.cvtColor(bgr, cv.COLOR_BGR2GRAY))
    if abs(ang) >= 0.5:
        bgr = rotate_bound(bgr, ang)

    # vorsichtig croppen (nur wenn sicher)
    bgr = safe_crop(bgr, conf=crop_conf)

    # Scan-Look
    if mode == "full":
        out = scan_full(bgr)
    elif mode == "soft":
        base = scan_soft(bgr)
        # Feintuning je nach gewünschter „Härte“
        if 0 <= bw_strength <= 1:
            # bw_strength 0.0 -> weicher; 1.0 -> härter
            mix = cv.cvtColor(cv.cvtColor(bgr, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
            out = cv.addWeighted(base, 0.5 + 0.5 * bw_strength, mix, 0.5 - 0.5 * bw_strength, 0)
        else:
            out = base
    else:
        out = scan_gentle(bgr)

    pdf_bytes = make_pdf(out, filename=filename, max_kb=max_kb, hard_cap_kb=hard_cap_kb)

    headers = {"Content-Disposition": f'inline; filename="{filename}"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
