import io
import base64
from typing import Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Query, Header
from fastapi.responses import JSONResponse
from PIL import Image
import img2pdf

app = FastAPI()


# ---------- Bild-Helfer ----------

def _read_upload_to_cv2(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=415, detail="Unsupported image format.")
    return img


def _order_points(pts: np.ndarray) -> np.ndarray:
    # sortiert 4 Punkte: [tl, tr, br, bl]
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
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
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped


def _detect_document_quad(bgr: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Sucht die größte 4-Eck-Kontur (Beleg) und gibt bei Erfolg die 4 Eckpunkte zurück.
    """
    img = bgr.copy()
    ratio = 1.0
    # kleiner skalieren für Stabilität/Geschwindigkeit
    max_side = 1200
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        ratio = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            # zurück auf Originalgröße mappen
            if ratio != 1.0:
                pts = pts / ratio
            return True, pts

    return False, np.zeros((0, 2), dtype="float32")


def _process_to_scan(bgr: np.ndarray) -> np.ndarray:
    """
    - Perspektive begradigen, falls möglich
    - in Graustufen + adaptives Threshold -> "Scan"-Look (Schwarz/Weiß)
    - leichte Kontrast-/Rausch-Optimierung
    Rückgabe: 8-bit 1-channel Bild (weiß=255, schwarz=0)
    """
    found, quad = _detect_document_quad(bgr)
    if found:
        bgr = _four_point_transform(bgr, quad)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # adaptive Schwelle für "gescannt"-Look
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25, 15
    )

    # kleine Artefakte glätten
    bw = cv2.medianBlur(bw, 3)

    return bw


def _pdf_bytes_from_ndarray(bw: np.ndarray) -> bytes:
    """
    Erzeugt ein einzelnes PDF aus einer OpenCV-Matrix.
    Nutzt img2pdf, erwartet RGB/Grayscale PIL-Image.
    """
    pil_img = Image.fromarray(bw)  # 'L' (8-bit gray)
    # einige Viewer mögen '1' (reines SW) – hier bleiben wir bei 'L' für Stabilität
    buf = io.BytesIO()
    # img2pdf erwartet bytes des Bildes
    with io.BytesIO() as img_buf:
        pil_img.save(img_buf, format="PNG")  # verlustfrei
        pdf = img2pdf.convert(img_buf.getvalue())
        buf.write(pdf)
    return buf.getvalue()


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/scan")
async def scan(
    file: UploadFile = File(...),
    response: str = Query(default="json", regex="^(json|pdf)$"),
    accept: str | None = Header(default=None)  # optionaler Header
):
    """
    POST /scan
    Form-Data: file (Bild)
    Query:
      - response=json (Default)  -> JSON mit base64 + data_url
      - response=pdf            -> direkte PDF (binary)
    Alternativ: Header `Accept: application/pdf` erzwingt die PDF-Antwort.
    """
    # Datei ins CV2-Image
    img_bgr = _read_upload_to_cv2(file)

    # "Scan" erzeugen
    bw = _process_to_scan(img_bgr)

    # PDF-Bytes bauen
    pdf_bytes = _pdf_bytes_from_ndarray(bw)

    wants_pdf = (response == "pdf") or (accept and "application/pdf" in accept.lower())

    if wants_pdf:
        # Direkt als Binary streamen (kein Speichern auf Server)
        headers = {"Content-Disposition": 'attachment; filename="scan.pdf"'}
        return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)

    # JSON-Variante für Zapier „Custom Request“
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    data_url = f"data:application/pdf;base64,{b64}"

    payload = {
        "filename": "scan.pdf",
        "content_type": "application/pdf",
        "file_base64": b64,
        "data_url": data_url,
        "note": "Use `data_url` directly in Zapier 'File' fields (Airtable, Drive, etc.). No decoding step needed."
    }
    return JSONResponse(payload)
