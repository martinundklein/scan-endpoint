from fastapi import FastAPI, HTTPException, Query, Response
import httpx, io, math, os
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2 as cv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Scan Endpoint")

# -----------------------------
# Utils
# -----------------------------
async def fetch_image(url: str) -> Image.Image:
    timeout = httpx.Timeout(20, read=60)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(400, f"file_url fetch failed: {r.status_code}")
        data = r.content
    img = Image.open(io.BytesIO(data))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")

def ensure_portrait(pil: Image.Image) -> Image.Image:
    w, h = pil.size
    if w > h:
        pil = pil.rotate(90, expand=True)
    return pil

def to_cv(pil: Image.Image) -> np.ndarray:
    return cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR)

def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))

# -----------------------------
# QR-Region maskieren
# -----------------------------
def mask_qr_like(gray: np.ndarray) -> np.ndarray:
    """
    Liefert Maske (uint8, 0/255), wo potenzielle QR-ähnliche Blöcke ausgeblendet werden.
    Heuristik: quadratische, sehr kontrastreiche, große Blöcke (10–40% der Breite),
    hohe Kanten-/Schachbrettdichte.
    """
    h, w = gray.shape
    # Kanten
    edges = cv.Canny(gray, 60, 160)
    # Morph zum Verbinden
    k = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    mask = np.ones_like(gray, dtype=np.uint8) * 255
    for c in contours:
        x, y, bw, bh = cv.boundingRect(c)
        area = bw * bh
        if area < (w*h)*0.02:  # zu klein
            continue
        ar = bw / max(bh, 1)
        # Quadrat-ish?
        if 0.8 <= ar <= 1.25:
            rel = bw / w
            if 0.12 <= rel <= 0.45:
                # “Gerastert”? Kanten-Dichte
                roi = edges[y:y+bh, x:x+bw]
                density = roi.mean() / 255.0
                if density > 0.25:
                    cv.rectangle(mask, (x, y), (x+bw, y+bh), 0, -1)
    return mask

# -----------------------------
# Deskew global
# -----------------------------
def deskew_global(gray: np.ndarray) -> np.ndarray:
    # leichte Blur gegen Rauschen
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(blur, 60, 160)
    lines = cv.HoughLines(edges, 1, np.pi/180, 160)
    angle = 0.0
    if lines is not None:
        angles = []
        for rho_theta in lines[:200]:
            rho, theta = rho_theta[0]
            deg = (theta * 180.0 / np.pi) - 90.0
            # nur leichte Schieflagen berücksichtigen
            if -15 <= deg <= 15:
                angles.append(deg)
        if len(angles) >= 3:
            angle = float(np.median(angles))
    if abs(angle) < 0.3:
        return gray
    # rotate around center
    h, w = gray.shape
    M = cv.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv.warpAffine(gray, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

# -----------------------------
# Belegdetektion (Quad + Score)
# -----------------------------
def find_receipt_quad(gray: np.ndarray, qr_mask: np.ndarray):
    h, w = gray.shape
    # Kanten unterdrücke QR
    work = cv.bitwise_and(gray, gray, mask=qr_mask)
    edges = cv.Canny(work, 60, 160)
    edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_RECT, (3,3)), iterations=1)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0

    for cnt in contours:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # Rechteckigkeit/Flächennutzung
        area = cv.contourArea(approx)
        if area < (w*h)*0.10:  # zu klein
            continue
        x, y, bw, bh = cv.boundingRect(approx)
        rect_area = bw * bh
        fill = area / max(rect_area, 1)

        # Seitenverhältnis
        ar = max(bw, bh) / max(min(bw, bh), 1)
        # Score komponieren
        size_score = min(1.0, area / (w*h*0.70))  # zu große (fast full-frame) eher schlechter
        ar_score = 1.0 if ar >= 2.2 else (ar / 2.2)  # Quittungen i.d.R. schlank
        rect_score = min(1.0, fill)  # je rechteckiger desto besser

        score = 0.45*rect_score + 0.35*ar_score + 0.20*size_score

        if score > best_score:
            best_score = score
            best = approx

    return best, float(best_score)

def warp_quad(img_bgr: np.ndarray, quad) -> np.ndarray:
    pts = quad.reshape(4,2).astype(np.float32)

    # sort -> (tl,tr,br,bl)
    def order(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl,tr,br,bl], dtype=np.float32)

    o = order(pts)
    (tl,tr,br,bl) = o
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(o, dst)
    warped = cv.warpPerspective(img_bgr, M, (maxW, maxH), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return warped

# -----------------------------
# B/W Scan Rendering
# -----------------------------
def bw_scan(pil: Image.Image, strength: float = 0.5) -> Image.Image:
    """Sauvola + sanftes Unsharp; strength 0..1"""
    gray = ImageOps.grayscale(pil)
    arr = np.array(gray)

    # Sauvola
    arrf = arr.astype(np.float32)
    # window 31, k=0.2 (mild)
    thresh = cv.ximgproc.niBlackThreshold(arr, 255, cv.THRESH_BINARY, 31, -0.2)
    bw = thresh

    # leicht entflecken
    bw = cv.medianBlur(bw, 3)

    out = Image.fromarray(bw)
    if strength > 0:
        out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(120*strength), threshold=2))
    return out

# -----------------------------
# PDF Builder
# -----------------------------
def build_pdf(pil_page: Image.Image) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4  # points

    # Bild nach A4 Breite/Höhe einpassen (Ränder 20pt)
    margin = 20
    avail_w = page_w - 2*margin
    avail_h = page_h - 2*margin
    img_w, img_h = pil_page.size
    scale = min(avail_w/img_w, avail_h/img_h)
    out_w = img_w*scale
    out_h = img_h*scale
    x = (page_w - out_w)/2
    y = (page_h - out_h)/2

    # Wichtig: ImageReader nutzen, sonst BytesIO-Fehler
    c.drawImage(ImageReader(pil_page), x, y, width=out_w, height=out_h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    return buf.getvalue()

# -----------------------------
# Main Endpoint (GET)
# -----------------------------
@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="publicly reachable image URL"),
    crop_threshold: float = Query(0.70, ge=0.0, le=1.0, description="confidence threshold 0..1"),
    bw_strength: float = Query(0.35, ge=0.0, le=1.0, description="B/W enhancement strength"),
):
    """
    Liefert immer ein PDF.
    - Cropt nur wenn Konfidenz >= crop_threshold.
    - Ignoriert große QR-ähnliche Blöcke bei der Belegdetektion.
    """
    # 1) Bild laden
    pil = await fetch_image(file_url)
    pil = ensure_portrait(pil)

    # 2) Vorbereitung
    bgr = to_cv(pil)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray = deskew_global(gray)

    # 3) QR maskieren & Beleg-Quad suchen
    qr_mask = mask_qr_like(gray)
    quad, score = find_receipt_quad(gray, qr_mask)

    # 4) ggf. croppen
    if quad is not None and score >= crop_threshold:
        warped = warp_quad(cv.cvtColor(gray, cv.COLOR_GRAY2BGR), quad)
        pil_warp = to_pil(warped)
        pil_warp = ensure_portrait(pil_warp)
        bw = bw_scan(pil_warp, strength=bw_strength)
    else:
        # kein Crop → sanfte B/W-Konvertierung auf Original
        bw = bw_scan(to_pil(cv.cvtColor(gray, cv.COLOR_GRAY2BGR)), strength=bw_strength)

    # 5) PDF erzeugen
    pdf_bytes = build_pdf(bw)

    headers = {
        "Content-Type": "application/pdf",
        "Content-Disposition": 'inline; filename="scan.pdf"',
        "X-Crop-Score": f"{score:.3f}" if quad is not None else "0.000",
        "X-Cropped": "1" if quad is not None and score >= crop_threshold else "0",
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
