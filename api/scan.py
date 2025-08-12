from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse, PlainTextResponse
import httpx
import io
import math
import numpy as np
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait
from reportlab.lib.utils import ImageReader

app = FastAPI(title="scan-endpoint")

# ------------------------- Utils -------------------------

async def fetch_image_bytes(url: str, timeout_s: float = 20.0) -> bytes:
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code != 200 or not r.content:
            raise HTTPException(status_code=400, detail=f"Bilddownload fehlgeschlagen ({r.status_code}).")
        return r.content

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    if img.mode not in ("RGB", "L", "RGBA"):
        img = img.convert("RGB")
    arr = np.array(img)
    if img.mode == "RGB":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if img.mode == "RGBA":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return arr  # L (grau)

def ensure_max_side(mat: np.ndarray, max_side: int = 2200) -> np.ndarray:
    h, w = mat.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return mat
    scale = max_side / float(s)
    return cv2.resize(mat, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def mask_qr(mat_bgr: np.ndarray) -> np.ndarray:
    # QR erkennen und weiß übermalen
    qr = cv2.QRCodeDetector()
    # detectAndDecode liefert bbox als 4 Punkte (1x4x2)
    data, points, _ = qr.detectAndDecode(mat_bgr)
    out = mat_bgr.copy()
    if points is not None and len(points) > 0:
        pts = points[0].astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        # etwas Puffer um den QR-Code
        pad = int(0.08 * max(w, h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(out.shape[1], x + w + pad)
        y1 = min(out.shape[0], y + h + pad)
        out[y0:y1, x0:x1] = 255  # weiß füllen
    return out

def find_doc_quad(mat_gray: np.ndarray) -> np.ndarray | None:
    # Canny + Konturen → größtes 4‑Punkt‑Polygon
    blur = cv2.GaussianBlur(mat_gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    h, w = mat_gray.shape[:2]
    best = None
    best_area = 0

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            # zu kleine Flächen ignorieren
            if area > best_area and area > 0.15 * (h * w):
                best = approx
                best_area = area

    if best is None:
        return None

    quad = best.reshape(4, 2).astype(np.float32)
    # sortiere Punkte: tl, tr, br, bl
    s = quad.sum(axis=1)
    diff = np.diff(quad, axis=1).ravel()
    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    tr = quad[np.argmin(diff)]
    bl = quad[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = pts
    # Seitenlängen
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0],
         [maxWidth - 1, 0],
         [maxWidth - 1, maxHeight - 1],
         [0, maxHeight - 1]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    return warped

def fallback_crop(mat_gray: np.ndarray) -> np.ndarray:
    # Minimaler Fallback: größtes Rotationsrechteck + Rand
    thresh = cv2.adaptiveThreshold(mat_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 10)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mat_gray
    big = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(big)  # (center,(w,h),angle)
    box = cv2.boxPoints(rect).astype(np.float32)
    warped = four_point_transform(mat_gray, order_quad(box))
    # leichter Rand weg
    h, w = warped.shape[:2]
    y0 = max(0, int(0.02 * h)); y1 = int(0.98 * h)
    x0 = max(0, int(0.04 * w)); x1 = int(0.96 * w)
    return warped[y0:y1, x0:x1]

def order_quad(quad: np.ndarray) -> np.ndarray:
    s = quad.sum(axis=1)
    diff = np.diff(quad, axis=1).ravel()
    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    tr = quad[np.argmin(diff)]
    bl = quad[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def deskew_to_portrait(img_gray: np.ndarray) -> np.ndarray:
    # binär für Winkelbestimmung
    bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bw == 0))  # schwarze Pixel
    if len(coords) < 100:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV: Winkel liegt in [-90,0)
    if angle < -45:
        angle = 90 + angle
    M = cv2.getRotationMatrix2D((img_gray.shape[1]//2, img_gray.shape[0]//2), angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (img_gray.shape[1], img_gray.shape[0]),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # auf Portrait drehen
    h, w = rotated.shape[:2]
    if w > h:
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
    return rotated

def sauvola_binarize(img_gray: np.ndarray) -> np.ndarray:
    # Sauvola via OpenCV: nehmen wir adaptiveGaussian als Näherung und verstärken Kontrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(img_gray)
    bw = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 12)
    return bw

def add_margin(img: np.ndarray, px: int = 24) -> np.ndarray:
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=255)

def make_pdf(bw_img: np.ndarray, filename: str = "scan.pdf") -> bytes:
    # PDF-Seitengröße passend zum Bild (72 dpi = pt): wir setzen ~ 96 dpi Skalierung für bessere Schärfe
    h, w = bw_img.shape[:2]
    dpi = 96.0
    page_w = w * 72.0 / dpi
    page_h = h * 72.0 / dpi

    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=(page_w, page_h))
    # als PNG in Memory für ImageReader
    png_bytes = cv2.imencode(".png", bw_img)[1].tobytes()
    img_ir = ImageReader(io.BytesIO(png_bytes))
    c.drawImage(img_ir, 0, 0, width=page_w, height=page_h)
    c.showPage()
    c.save()
    return buff.getvalue()

# ------------------------- Pipeline -------------------------

def process_image_to_bw(mat_bgr: np.ndarray) -> np.ndarray:
    # 1) Resize
    mat_bgr = ensure_max_side(mat_bgr, 2400)

    # 2) QR maskieren
    mat_bgr = mask_qr(mat_bgr)

    # 3) Graustufen
    gray = cv2.cvtColor(mat_bgr, cv2.COLOR_BGR2GRAY)

    # 4) Dokument finden
    quad = find_doc_quad(gray)
    if quad is not None:
        warped = four_point_transform(gray, quad)
    else:
        warped = fallback_crop(gray)

    # 5) Begradigen & Portrait
    warped = deskew_to_portrait(warped)

    # 6) Binarisieren
    bw = sauvola_binarize(warped)

    # 7) leichter Rand
    bw = add_margin(bw, 18)

    return bw

# ------------------------- Routes -------------------------

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"

@app.get("/scan")
async def scan(file_url: str):
    """
    Beispiel:
      GET /scan?file_url=https://.../bild.jpg
    Antwort:
      application/pdf (Dateistream), Content-Disposition: attachment; filename=scan.pdf
    """
    # 1) Bild bytes holen
    img_bytes = await fetch_image_bytes(file_url)

    # 2) PIL laden
    try:
        pil = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ungültiges Bildformat: {e}")

    # 3) PIL -> OpenCV
    mat = pil_to_cv2(pil)
    if mat is None:
        raise HTTPException(status_code=400, detail="Bildkonvertierung fehlgeschlagen.")

    # 4) Pipeline
    try:
        bw = process_image_to_bw(mat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verarbeitung fehlgeschlagen: {e}")

    # 5) PDF bauen
    try:
        pdf_bytes = make_pdf(bw, "scan.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF-Erzeugung fehlgeschlagen: {e}")

    headers = {
        "Content-Disposition": 'attachment; filename="scan.pdf"'
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
