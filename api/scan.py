from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse, Response, PlainTextResponse
from pydantic import BaseModel
import httpx, asyncio, io, math, time
import numpy as np
import cv2
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Scan Endpoint", version="2.0.0")

# ---------- HTTP Client (shared) ----------
TIMEOUT = httpx.Timeout(30.0, connect=15.0)
CLIENT = httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True)

# ---------- Models ----------
class ScanIn(BaseModel):
    file_url: str

# ---------- Utilities ----------
def _bytes_to_cv2_image(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Bild konnte nicht geladen werden (unterstütztes Format: jpg/png).")
    return img

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    # Breite/Höhe
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
    warped = cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_CUBIC)
    return warped

def _deskew(gray: np.ndarray) -> np.ndarray:
    # Schätzung anhand der größten Text-Kanten
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 150)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for rho, theta in lines[:,0]:
            a = (theta * 180 / math.pi) % 180
            if 10 < a < 170:  # horizontale/vertikale ausschließen
                if a > 90:
                    a = a - 180
                angles.append(a)
        if len(angles) > 0:
            angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return gray
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def _find_document_contour(gray: np.ndarray) -> np.ndarray | None:
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2).astype("float32")
    return None

def _to_scanned(gray: np.ndarray) -> np.ndarray:
    # adaptive Threshold für "gescannt"-Look
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 15
    )
    # leichtes Öffnen, um Rauschen zu entfernen
    kernel = np.ones((2,2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return th

def _add_white_margin(img: np.ndarray, px: int = 24) -> np.ndarray:
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=[255,255,255])

def _np_bw_to_pil(img_bw: np.ndarray) -> Image.Image:
    if len(img_bw.shape) == 2:
        pil = Image.fromarray(img_bw)
    else:
        pil = Image.fromarray(cv2.cvtColor(img_bw, cv2.COLOR_BGR2RGB))
    return pil

def _pil_to_pdf_bytes(pil_img: Image.Image, dpi: int = 200) -> bytes:
    # in ReportLab-Seite einbetten (letter oder passend skalieren)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter

    img_w, img_h = pil_img.size
    # Zielbreite: 90% der Seite
    scale = min(page_w * 0.9 / img_w, page_h * 0.9 / img_h)
    new_w, new_h = img_w * scale, img_h * scale
    x = (page_w - new_w) / 2
    y = (page_h - new_h) / 2

    c.drawImage(ImageReader(pil_img), x, y, width=new_w, height=new_h)
    c.showPage()
    c.save()
    return buf.getvalue()

async def _download(url: str) -> bytes:
    r = await CLIENT.get(url)
    r.raise_for_status()
    return r.content

# -------- Uploaders (with fallback) --------
async def _upload_tmpfiles(pdf_bytes: bytes, filename: str) -> str | None:
    try:
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        r = await CLIENT.post("https://tmpfiles.org/api/v1/upload", files=files)
        r.raise_for_status()
        js = r.json()
        # API liefert z.B. {"status":true,"data":{"url":"https://tmpfiles.org/dl/<id>/scan.pdf"}}
        url = (js.get("data") or {}).get("url")
        return url
    except Exception:
        return None

async def _upload_transfersh(pdf_bytes: bytes, filename: str) -> str | None:
    try:
        r = await CLIENT.post(f"https://transfer.sh/{filename}", content=pdf_bytes,
                              headers={"Content-Type": "application/pdf"})
        if r.status_code == 200 and r.text.startswith("https://"):
            return r.text.strip()
        return None
    except Exception:
        return None

async def _upload_fileio(pdf_bytes: bytes, filename: str) -> str | None:
    try:
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        r = await CLIENT.post("https://file.io", files=files)
        r.raise_for_status()
        js = r.json()
        # {"success":true,"link":"https://file.io/xxxx","expiry":"14 days"}
        if js.get("success") and js.get("link"):
            return js["link"]
        return None
    except Exception:
        return None

async def upload_with_fallback(pdf_bytes: bytes, filename: str) -> str:
    for fn in (_upload_tmpfiles, _upload_transfersh, _upload_fileio):
        url = await fn(pdf_bytes, filename)
        if url:
            return url
    raise RuntimeError("Kein Upload-Dienst erreichbar (tmpfiles → transfer.sh → file.io).")

# -------- Core Processing --------
def process_image_to_pdf(image_bgr: np.ndarray) -> bytes:
    # 1) Grundwandlung
    orig = image_bgr.copy()
    # leichte Kontrastanhebung
    lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l,a,b))
    orig = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # 2) Deskew (falls nötig)
    gray = _deskew(gray)

    # 3) Dokument suchen (4-Punkt)
    quad = _find_document_contour(gray)
    if quad is not None:
        warped = _four_point_transform(orig, quad)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 4) "Scan"-Look
    bw = _to_scanned(gray)
    bw = _add_white_margin(bw, 30)

    # 5) nach Portrait drehen (falls höher als breit ist ok; wenn quer, ggf. drehen)
    h, w = bw.shape[:2]
    if w > h * 1.1:
        bw = cv2.rotate(bw, cv2.ROTATE_90_CLOCKWISE)

    # 6) PDF erzeugen
    pil = _np_bw_to_pil(bw)
    pdf_bytes = _pil_to_pdf_bytes(pil)
    return pdf_bytes

# -------- Endpoints --------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK - /scan?file_url=ENCODED_URL&response=link|pdf"

@app.get("/scan")
async def scan_get(
    file_url: str = Query(..., description="Öffentliche Bild-URL (Airtable Signed URL etc.)"),
    response: str = Query("link", pattern="^(link|pdf)$")
):
    """
    GET-Variante (einfach für Zapier Webhooks):
    /scan?file_url=<urlencodete_url>&response=link|pdf
    """
    img_bytes = await _download(file_url)
    img = _bytes_to_cv2_image(img_bytes)
    pdf_bytes = process_image_to_pdf(img)
    filename = "scan.pdf"

    if response == "pdf":
        return Response(content=pdf_bytes, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{filename}"'})
    else:
        url = await upload_with_fallback(pdf_bytes, filename)
        return JSONResponse({"filename": filename, "url": url})

@app.post("/scan")
async def scan_post(
    payload: ScanIn = Body(...),
    response: str = Query("link", pattern="^(link|pdf)$")
):
    """
    POST-Variante (JSON):
    { "file_url": "https://..." }
    """
    img_bytes = await _download(payload.file_url)
    img = _bytes_to_cv2_image(img_bytes)
    pdf_bytes = process_image_to_pdf(img)
    filename = "scan.pdf"

    if response == "pdf":
        return Response(content=pdf_bytes, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{filename}"'})
    else:
        url = await upload_with_fallback(pdf_bytes, filename)
        return JSONResponse({"filename": filename, "url": url})

# ---------- Shutdown cleanup ----------
@app.on_event("shutdown")
async def _shutdown():
    try:
        await CLIENT.aclose()
    except Exception:
        pass
