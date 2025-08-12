import io
import uuid
import math
import httpx
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from PIL import Image, ImageOps, ImageEnhance, ImageStat, ImageFilter, ExifTags
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# OpenCV optional (für Render: opencv-python-headless in requirements.txt)
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

app = FastAPI(title="Scan API", description="Scan: straighten, crop, binarize → PDF", version="2.0")

# ------------------------------------------------------------
# HTTP / MIME
# ------------------------------------------------------------

async def fetch_bytes(url: str):
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "")

def looks_like_pdf(ctype: str, data: bytes) -> bool:
    return (ctype and "pdf" in ctype.lower()) or data.startswith(b"%PDF")

# ------------------------------------------------------------
# PIL Utilities
# ------------------------------------------------------------

def exif_transpose(img: Image.Image) -> Image.Image:
    # Korrigiert iPhone/Android Rotationsflag
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def enhance_and_binarize_pil(img: Image.Image) -> Image.Image:
    """Einfacher Fallback: Kontrast hoch, dann Otsu-ähnlich per Mittelwert."""
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    arr = np.array(gray)
    thresh = arr.mean()
    bw = (arr > thresh).astype(np.uint8) * 255
    return Image.fromarray(bw)

# ------------------------------------------------------------
# OpenCV: Dokument finden, entzerren, binarisieren, crop
# ------------------------------------------------------------

def order_pts(pts):
    # Sortiert 4 Punkte in Reihenfolge: TL, TR, BR, BL
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image, pts):
    rect = order_pts(pts)
    (tl, tr, br, bl) = rect
    # Zielbreite/-höhe
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped

def auto_crop_border(bw_np: np.ndarray, pad: int = 6) -> np.ndarray:
    # Entfernt weiße Ränder: finde Bounding Box aller „schwarzen“ Pixel
    # bw_np ist 0/255
    mask = (bw_np < 250).astype(np.uint8)  # alles was nicht fast weiß ist
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return bw_np  # nichts gefunden
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    y_min = max(0, y_min - pad); x_min = max(0, x_min - pad)
    y_max = min(bw_np.shape[0]-1, y_max + pad); x_max = min(bw_np.shape[1]-1, x_max + pad)
    return bw_np[y_min:y_max+1, x_min:x_max+1]

def cv2_scan(img_pil: Image.Image) -> Image.Image:
    """
    Voller OpenCV-Pipeline-Scan:
    - Kanten finden
    - größtes 4-Eck (Dokument) wählen
    - Perspektive entzerren
    - adaptives Threshold (Scan-Look)
    - Rand automatisch croppen
    Fallback: falls 4-Eck nicht gefunden → nur B/W.
    """
    if not HAS_CV2:
        return enhance_and_binarize_pil(img_pil)

    # PIL → OpenCV (BGR)
    img = np.array(img_pil.convert("RGB"))
    img = img[:, :, ::-1]  # RGB → BGR

    # kleiner Vorschau-Frame für robustere Konturen
    h, w = img.shape[:2]
    scale = 800.0 / max(h, w)
    scale = min(1.0, scale)
    small = cv2.resize(img, (int(w*scale), int(h*scale)))
    ratio = 1.0 / scale

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kanten → Konturen
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    screenCnt = None
    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx.reshape(4, 2) * ratio
            break

    if screenCnt is None:
        # kein sauberes 4‑Eck gefunden → Fallback
        return enhance_and_binarize_pil(img_pil)

    # Perspektive entzerren (auf Originalgröße)
    warped = four_point_transform(img, screenCnt.astype("float32"))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # adaptives Threshold → „Scan“-Look
    bw = cv2.adaptiveThreshold(
        warped_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    # Auto-Cropping von weißen Rändern
    bw = auto_crop_border(bw, pad=8)

    # Zurück nach PIL
    result = Image.fromarray(bw)
    return result

# ------------------------------------------------------------
# PDF-Erstellung
# ------------------------------------------------------------

def make_pdf_from_image(img: Image.Image) -> bytes:
    """
    Legt das Bild mittig auf eine A4-Seite. (Kein Upscaling über A4 hinaus)
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    a4_w, a4_h = A4

    img_bytes = io.BytesIO()
    # Für „saubere“ Kanten im PDF: als PNG speichern
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    iw, ih = img.size
    ratio = min(a4_w / iw, a4_h / ih)
    new_w, new_h = iw * ratio, ih * ratio
    x = (a4_w - new_w) / 2
    y = (a4_h - new_h) / 2

    c.drawImage(ImageReader(img_bytes), x, y, width=new_w, height=new_h, preserveAspectRatio=True, mask="auto")
    c.showPage()
    c.save()
    return buf.getvalue()

# ------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------

@app.get("/")
def root():
    return {"ok": True, "msg": "Use /scan?file_url=... (returns PDF)"}

@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Direkter Link zur Bild- oder PDF-Datei"),
    response: str = Query("pdf", description="'pdf' für direkte Datei-Ausgabe")
):
    raw, ctype = await fetch_bytes(file_url)

    # Wenn bereits PDF: einfach durchreichen (keine Bildverbesserung)
    if looks_like_pdf(ctype, raw):
        pdf_bytes = raw
    else:
        # Bild laden + EXIF-Rotation korrigieren
        img = Image.open(io.BytesIO(raw))
        img = exif_transpose(img)

        # Querformat → optional automatisch drehen, damit Hochformat bevorzugt ist
        if img.width > img.height:
            img = img.rotate(90, expand=True)

        # OpenCV-Pipeline (mit Fallback)
        scanned = cv2_scan(img)
        pdf_bytes = make_pdf_from_image(scanned)

    # Immer PDF direkt zurückgeben
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="scan.pdf"'}
    )
