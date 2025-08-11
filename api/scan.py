from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import requests
import io
import base64
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from PIL import Image

app = FastAPI()

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # Reihenfolge: tl, tr, br, bl
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect

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
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped

def to_scanned_bw(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # etwas Glättung gegen Rauschen, aber Kanten behalten
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # „Scanner-Look“
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )
    return bw

def detect_document(image: np.ndarray) -> np.ndarray:
    # verkleinern zum Finden, Ratio merken
    ratio = image.shape[0] / 700.0
    small = cv2.resize(image, (int(image.shape[1] / ratio), 700))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32") * ratio
            return four_point_transform(image, pts)

    # Fallback: kein Viereck erkannt → Original zurück
    return image

def pdf_bytes_from_bw(bw_np: np.ndarray) -> bytes:
    h, w = bw_np.shape[:2]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    # OpenCV -> PIL -> ReportLab
    pil = Image.fromarray(bw_np).convert("L").convert("RGB")
    img = ImageReader(pil)

    page_w, page_h = A4
    margin = 24
    scale = min((page_w - 2 * margin) / w, (page_h - 2 * margin) / h)
    draw_w, draw_h = w * scale, h * scale
    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2

    c.drawImage(img, x, y, draw_w, draw_h)
    c.showPage()
    c.save()
    return buf.getvalue()

@app.post("/scan")
async def scan(req: Request):
    try:
        data = await req.json()
    except Exception:
        return Response("Invalid JSON", status_code=400)

    url = data.get("file_url")
    if not url:
        return Response("Missing file_url", status_code=400)

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return Response(f"Failed to fetch image: {e}", status_code=400)

    img_bytes = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return Response("Unsupported image", status_code=415)

    # Dokument finden & begradigen
    doc = detect_document(img)
    # Schwarz/Weiß „Scan“
    bw = to_scanned_bw(doc)
    # PDF erzeugen
    pdf = pdf_bytes_from_bw(bw)

    # Base64 zurückgeben (keine Speicherung)
    b64 = base64.b64encode(pdf).decode("utf-8")
    return JSONResponse({
        "filename": "scan.pdf",
        "content_type": "application/pdf",
        "file_base64": b64
    })
