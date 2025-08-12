from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import io
import numpy as np
import cv2 as cv
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, portrait
from reportlab.lib.utils import ImageReader
import httpx
from typing import Tuple, List, Optional

app = FastAPI(title="Scan Endpoint")

# ---------- Hilfsfunktionen ----------

async def fetch_bytes(url: str, timeout: float = 30.0) -> bytes:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code != 200 or not r.content:
            raise HTTPException(400, detail=f"Download fehlgeschlagen ({r.status_code})")
        return r.content

def bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, detail="Bild konnte nicht decodiert werden")
    return img

def to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def mask_qr_regions(gray: np.ndarray) -> np.ndarray:
    """Findet QR-Codes und maskiert deren Bounding-Boxen im Bild (setzt sie auf Mittelgrau),
       damit sie bei der Dokument-Erkennung nicht dominieren."""
    h, w = gray.shape
    detector = cv.QRCodeDetector()
    # OpenCV >=4.7: detectAndDecodeMulti liefert Punkte; fallback auf single
    ok, points = detector.detect(gray)
    out = gray.copy()
    if ok and points is not None:
        pts = points.astype(int)
        x, y, ww, hh = cv.boundingRect(pts)
        cv.rectangle(out, (x, y), (x+ww, y+hh), int(np.median(gray)), thickness=-1)
        return out

    try:
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(gray)
        if retval and points is not None:
            for quad in points:
                x, y, ww, hh = cv.boundingRect(quad.astype(int))
                cv.rectangle(out, (x, y), (x+ww, y+hh), int(np.median(gray)), thickness=-1)
    except Exception:
        pass
    return out

def order_quad(pts: np.ndarray) -> np.ndarray:
    """Sortiert 4 Eckpunkte (tl, tr, br, bl)"""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def perspective_crop(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    quad = order_quad(quad.astype(np.float32))
    (tl, tr, br, bl) = quad
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    maxW = max(maxW, 100)
    maxH = max(maxH, 100)

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(quad, dst)
    warped = cv.warpPerspective(bgr, M, (maxW, maxH), flags=cv.INTER_CUBIC)
    return warped

def doc_detect(bgr: np.ndarray, qr_masked_gray: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Sucht das Dokument per Konturen. Gibt (Quad, confidence) zurück.
       Confidence basiert auf Fläche, Rechteckigkeit und Seitenverhältnis."""
    gray = qr_masked_gray
    h, w = gray.shape

    # Kantenrobust: leicht glätten + Canny + Morph
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blur, 50, 150)
    edges = cv.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((5,5), np.uint8))

    cnts, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best_conf = 0.0
    best_quad = None
    img_area = float(h*w)

    for c in cnts:
        area = cv.contourArea(c)
        if area < img_area * 0.10:   # ignoriere Kleinkram
            continue
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # Rechteckigkeit: Verhältnis Konturfläche / Bounding-Rect-Fläche
        x,y,ww,hh = cv.boundingRect(approx)
        rect_area = float(ww*hh) if ww*hh>0 else 1.0
        rectangularity = float(area) / rect_area

        # Coverage: wieviel des Bildes wird belegt?
        coverage = area / img_area

        # Seitenverhältnis „Beleg-like“ (z.B. 1:2 bis 1:5)
        ar = max(ww,hh)/max(1.0,min(ww,hh))
        if ar < 1.2:   # sehr quadratisch → abwerten (QR etc.)
            ar_score = 0.2
        elif ar < 1.6:
            ar_score = 0.6
        elif ar < 3.5:
            ar_score = 1.0
        elif ar < 6.0:
            ar_score = 0.7
        else:
            ar_score = 0.4

        conf = (0.45*rectangularity + 0.45*coverage + 0.10*ar_score)
        if conf > best_conf:
            best_conf = conf
            best_quad = approx.reshape(-1,2)

    return best_quad, float(best_conf)

def soft_bw(bgr: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Sanfte Schwarz/Weiß-Optik (nicht zu hart)."""
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    # Kontrast anheben (CLAHE)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Leichte Entzerrung der Ausleuchtung
    gray = cv.GaussianBlur(gray, (3,3), 0)
    # Adaptive Threshold (weicher), dann mit Original mischen
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, 31, 8)
    # Stärke 0..1 -> wie stark Richtung reines B/W
    if strength < 0: strength = 0
    if strength > 1: strength = 1
    mix = cv.addWeighted(gray, 1.0-strength, th, strength, 0)
    # In 3-Kanal für PDF
    return cv.cvtColor(mix, cv.COLOR_GRAY2BGR)

def npimage_to_pdf_bytes(bgr: np.ndarray, filename: str = "scan.pdf") -> bytes:
    """Erzeugt ein 1‑seitiges PDF mit dem Bild; Seite passend zum Bild skaliert (A4 Hochformat, zentriert)."""
    pil = to_pil(bgr)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=portrait(A4))
    pw, ph = portrait(A4)   # Punkte (72 DPI)
    # Bildgröße im PDF so skalieren, dass es mit 10 mm Rand passt
    margin = 28.35  # 10 mm in pt
    avail_w = pw - 2*margin
    avail_h = ph - 2*margin
    img_w, img_h = pil.size
    scale = min(avail_w/img_w, avail_h/img_h)
    out_w = img_w * scale
    out_h = img_h * scale
    x = (pw - out_w) / 2.0
    y = (ph - out_h) / 2.0
    # Wichtig: ImageReader benutzen, nicht BytesIO direkt
    c.drawImage(ImageReader(pil), x, y, width=out_w, height=out_h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf

# ---------- API ----------

@app.get("/", response_class=JSONResponse)
def root():
    return {"ok": True, "service": "scan-endpoint", "endpoints": ["/scan"]}

@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Direkter Bild-URL (JPG/PNG)"),
    crop_confidence: float = Query(0.7, ge=0.0, le=1.0, description="Ab welcher Sicherheit wird zugeschnitten/entzerrt"),
    bw_strength: float = Query(0.55, ge=0.0, le=1.0, description="Intensität der B/W-Optik (0..1)")
):
    # 1) Eingabe holen
    img_bytes = await fetch_bytes(file_url)
    bgr = bytes_to_bgr(img_bytes)

    # 2) QR‑Codes maskieren und Dokument suchen
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray_qrmasked = mask_qr_regions(gray)
    quad, conf = doc_detect(bgr, gray_qrmasked)

    # 3) Bild vorbereiten
    if quad is not None and conf >= crop_confidence:
        cropped = perspective_crop(bgr, quad)
        work = cropped
    else:
        work = bgr

    # 4) Weiche B/W‑Optik anwenden
    bw_img = soft_bw(work, strength=bw_strength)

    # 5) PDF erzeugen
    pdf_bytes = npimage_to_pdf_bytes(bw_img, filename="scan.pdf")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="scan.pdf"'}
    )
