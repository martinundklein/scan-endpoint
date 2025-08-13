# api/scan.py
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
import httpx
import numpy as np
import cv2 as cv
from PIL import Image
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Scan Endpoint")

# --------- Hilfen ---------

async def fetch_image_bytes(url: str, timeout_s: float = 20.0) -> bytes:
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content

def imdecode_rgb(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("Bild konnte nicht dekodiert werden")
    return img

def detect_doc_quad(img_bgr: np.ndarray, min_conf: float = 0.65):
    """Finde 4‑Eck der Quittung. Liefert None, wenn unsicher."""
    h, w = img_bgr.shape[:2]
    scale = 1200.0 / max(h, w) if max(h, w) > 1200 else 1.0
    img_small = cv.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 7, 50, 50)
    edges = cv.Canny(gray, 50, 150)

    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:8]

    for cnt in contours:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv.contourArea(approx)
            img_area = img_small.shape[0] * img_small.shape[1]
            conf = area / float(img_area)
            if conf >= min_conf:
                # auf Originalkoordinaten hochskalieren
                quad = (approx.reshape(4, 2) / scale).astype(np.float32)
                return order_points(quad)
    return None

def order_points(pts: np.ndarray) -> np.ndarray:
    """Sortiere 4 Punkte (tl, tr, br, bl)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def warp_perspective(img_bgr: np.ndarray, rect: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(rect, dst)
    return cv.warpPerspective(img_bgr, M, (maxW, maxH), flags=cv.INTER_CUBIC)

def nice_bw_scan(img_bgr: np.ndarray) -> np.ndarray:
    """Sanfter Scan‑Look: Hintergrund glätten, Text dunkel, nicht zu hart."""
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    # Hintergrund-Normalisierung
    # (grobe Beleuchtung entfernen, ohne Text zu killen)
    bg = cv.GaussianBlur(gray, (0, 0), 21)
    norm = cv.divide(gray, bg, scale=180)  # 180 = mittlerer Pegel

    # sanftes adaptives Thresholding
    th = cv.adaptiveThreshold(
        norm, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 10
    )

    # leichte Entzerrung harter Pixel
    th = cv.medianBlur(th, 3)
    return th  # uint8, 0/255

def to_pil_gray(arr_u8: np.ndarray) -> Image.Image:
    if arr_u8.ndim == 2:
        return Image.fromarray(arr_u8, mode="L")
    return Image.fromarray(cv.cvtColor(arr_u8, cv.COLOR_BGR2RGB)).convert("L")

def build_pdf_from_pil(
    pil_img: Image.Image,
    target_kb: int = 400,
    max_kb: int = 800,
    start_quality: int = 60,
    dpi: int = 144,
) -> bytes:
    """
    Erzeugt PDF mit JPEG‑komprimiertem Graustufenbild.
    Versucht Quality/Skalierung zu reduzieren, bis Zielgröße erreicht ist.
    """
    scale = 1.0
    quality = start_quality

    for _ in range(12):  # maximal 12 Versuche
        # optional skalieren
        if scale < 0.999:
            new_w = max(400, int(pil_img.width * scale))
            new_h = max(400, int(pil_img.height * scale))
            work = pil_img.resize((new_w, new_h), Image.LANCZOS)
        else:
            work = pil_img

        # JPEG in Memory
        jpg_buf = BytesIO()
        work.save(jpg_buf, format="JPEG", quality=quality, optimize=True)
        jpg_buf.seek(0)

        # PDF bauen
        w_pt = work.width * 72.0 / dpi
        h_pt = work.height * 72.0 / dpi
        pdf_buf = BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=(w_pt, h_pt))
        c.drawImage(ImageReader(jpg_buf), 0, 0, width=w_pt, height=h_pt, preserveAspectRatio=True, mask='auto')
        c.showPage()
        c.save()
        pdf_bytes = pdf_buf.getvalue()
        size_kb = len(pdf_bytes) / 1024.0

        if size_kb <= target_kb:
            return pdf_bytes

        # noch zu groß → zuerst Qualität senken, dann skalieren
        if quality > 35:
            quality -= 7
        else:
            scale *= 0.9
            if size_kb <= max_kb or scale < 0.55:
                return pdf_bytes  # akzeptieren, sonst würde es zu klein

    return pdf_bytes

# --------- API ---------

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"

@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Direkte Bild-URL (Airtable Attachment URL)"),
    min_conf: float = Query(0.65, ge=0.0, le=1.0, description="Mindest-Confidence für Begradigung"),
    target_kb: int = Query(400, ge=50, description="Zielgröße in KB"),
    max_kb: int = Query(800, ge=100, description="harte Obergrenze in KB"),
):
    try:
        raw = await fetch_image_bytes(file_url)
        img = imdecode_rgb(raw)

        # vorsichtige Begradigung (nur wenn sicher genug)
        quad = detect_doc_quad(img, min_conf=min_conf)
        if quad is not None:
            img = warp_perspective(img, quad)

        # Scan‑Look
        bw = nice_bw_scan(img)

        # nach sehr langen Kanten für typische Belege richten (hochkant)
        h, w = bw.shape[:2]
        if w > h and (w / max(1, h)) > 1.4:
            # sehr breite Aufnahme → ggf. rotieren
            bw = cv.rotate(bw, cv.ROTATE_90_COUNTERCLOCKWISE)

        pil = to_pil_gray(bw)
        pdf_bytes = build_pdf_from_pil(pil, target_kb=target_kb, max_kb=max_kb)

        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": 'inline; filename="scan.pdf"'},
        )

    except httpx.HTTPError as e:
        return PlainTextResponse(f"Download fehlgeschlagen: {str(e)}", status_code=400)
    except Exception as e:
        return PlainTextResponse(f"Verarbeitung fehlgeschlagen: {str(e)}", status_code=400)
