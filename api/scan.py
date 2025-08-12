# api/scan.py
import io
from typing import Optional, Tuple, List

import cv2
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

app = FastAPI(title="scan-endpoint")

# ---------------- IO ----------------

def fetch_image_bytes(url: str) -> bytes:
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download fehlgeschlagen: {e}")

def imread_color(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=415, detail="Bildformat wird nicht unterstützt")
    return img

# ---------------- Utility ----------------

def to_portrait(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if w > h else img

def deskew_if_confident(bw: np.ndarray) -> np.ndarray:
    # Hough-Linien, um einen dominanten Winkel zu schätzen
    lines = cv2.HoughLines(bw, 1, np.pi/180, 220)
    if lines is None:
        return bw
    angles = []
    for rho, theta in lines[:, 0]:
        ang = np.degrees(theta)
        if ang > 90:
            ang -= 180
        if -60 < ang < 60:
            angles.append(ang)
    if not angles:
        return bw
    angle = np.median(angles)
    # „Sicher“ nur, wenn |Winkel| < 6° und genügend Linien
    if abs(angle) < 6 and len(angles) >= 10:
        h, w = bw.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    return bw

def suppress_qr_regions(gray: np.ndarray) -> np.ndarray:
    """
    Unterdrückt große, hochfrequente, nahezu quadratische Bereiche (QR/DataMatrix),
    damit diese die Kontursuche nicht dominieren.
    """
    edges = cv2.Canny(gray, 60, 180)
    dil = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)), 1)
    open15 = cv2.morphologyEx(dil, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15,15)))
    cnts, _ = cv2.findContours(open15, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    mask = np.zeros_like(gray)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < 0.03*H*W:  # nur wirklich große Blöcke
            continue
        ar = max(w,h)/max(1,min(w,h))
        if 0.8 <= ar <= 1.25:  # ziemlich quadratisch → sehr wahrscheinlich QR
            cv2.rectangle(mask,(x,y),(x+w,y+h),255,-1)
    if mask.max() > 0:
        mask = cv2.GaussianBlur(mask,(31,31),0)
        out = gray.copy()
        out[mask>0] = np.minimum(255, out[mask>0] + 90)  # „aufhellen“
        return out
    return gray

# ---------------- Beleg-Erkennung ----------------

def find_receipt_quad(img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    Liefert (vier Punkte), Score (0..1).
    Score kombiniert: Flächenanteil, Rechteckigkeit, Seitenverhältnis, Kantendichte.
    """
    h, w = img_bgr.shape[:2]
    scale = 1600 / max(h, w) if max(h, w) > 1600 else 1.0
    small = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    gray = suppress_qr_regions(gray)

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)),1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0

    best = None
    best_score = 0.0
    img_area = small.shape[0]*small.shape[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) != 4:
            continue
        area = abs(cv2.contourArea(approx))
        if area < 0.15*img_area:  # recht groß
            continue

        # Rechteckigkeit & Seitenverhältnis
        rect = cv2.minAreaRect(approx)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0:
            continue
        ar = max(rw, rh)/min(rw, rh)

        # Quittungen sind meistens deutlich höher als breit
        ar_score = np.clip((ar-1.6)/7.0, 0, 1)  # ab ~1.6 aufwärts, bis ~8 ganz gut

        # Flächenanteil
        area_score = np.clip((area/img_area - 0.15)/0.6, 0, 1)

        # Rechteck-Abweichung (Solidity)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box.astype(np.float32))
        rect_score = np.clip(area/max(1.0, box_area), 0, 1)

        score = 0.45*area_score + 0.35*ar_score + 0.20*rect_score
        if score > best_score:
            best_score = score
            best = approx

    if best is None:
        return None, 0.0

    # zurück auf Originalgröße skalieren
    M = np.array([[1/scale,0],[0,1/scale]], dtype=np.float32)
    best = (best.reshape(-1,2).astype(np.float32) @ M.T).reshape(-1,1,2)
    return best, float(best_score)

# ---------------- Binarisierung + PDF ----------------

def binarize_scan(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 60, 60)
    # leichte Kontrastanhebung
    g = cv2.convertScaleAbs(g, alpha=1.15, beta=0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 15)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),1)
    return bw

def to_pdf_bytes(bw: np.ndarray) -> bytes:
    a4_w, a4_h = A4
    # zu Portrait
    if bw.shape[1] > bw.shape[0]:
        bw = cv2.rotate(bw, cv2.ROTATE_90_CLOCKWISE)
    rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    ok, png = cv2.imencode(".png", rgb)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG-Encoding fehlgeschlagen")
    bio_png = io.BytesIO(png.tobytes())
    img_reader = ImageReader(bio_png)

    # etwas Rand
    mm = 72/25.4
    margin = int(12*mm)
    max_w = int(a4_w - 2*margin)
    max_h = int(a4_h - 2*margin)
    ih, iw = bw.shape[:2]
    scale = min(max_w/iw, max_h/ih)
    draw_w = int(iw*scale)
    draw_h = int(ih*scale)
    x = int((a4_w - draw_w)/2)
    y = int((a4_h - draw_h)/2)

    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    c.drawImage(img_reader, x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    return out.getvalue()

# ---------------- Endpoints ----------------

@app.get("/")
def root():
    return JSONResponse({"ok": True, "usage": "/scan?file_url=<öffentliche Bild-URL>"})

@app.get("/scan")
def scan(file_url: str = Query(..., description="Öffentliche Bild-URL (z.B. Airtable Attachment URL)")):
    # 1) Laden
    raw = fetch_image_bytes(file_url)
    img = imread_color(raw)

    # 2) Versuch: Belegerkennung
    quad, score = find_receipt_quad(img)

    # 3) Pipeline je nach Sicherheit
    if quad is not None and score >= 0.90:
        # sicher → entzerren, dann binarisieren
        warped = four_point_transform(img, quad)
        warped = to_portrait(warped)
        bw = binarize_scan(warped)
    else:
        # nicht sicher → NICHT croppen, nur B/W + ggf. leichte Entzerrung
        img_p = to_portrait(img)
        bw = binarize_scan(img_p)
        bw = deskew_if_confident(bw)

    # 4) PDF
    pdf = to_pdf_bytes(bw)
    headers = {"Content-Disposition": 'inline; filename="scan.pdf"'}
    return Response(content=pdf, media_type="application/pdf", headers=headers)

# --------- Geometrie-Funktionen (für sicheres Cropping) ---------

def order_pts(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4,2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl,tr,br,bl], dtype="float32")

def four_point_transform(img: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    rect = order_pts(cnt)
    (tl,tr,br,bl) = rect
    wA = np.hypot(br[0]-bl[0], br[1]-bl[1])
    wB = np.hypot(tr[0]-tl[0], tr[1]-tl[1])
    hA = np.hypot(tr[0]-br[0], tr[1]-br[1])
    hB = np.hypot(tl[0]-bl[0], tl[1]-bl[1])
    maxW = int(max(wA,wB))
    maxH = int(max(hA,hB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_CUBIC)
