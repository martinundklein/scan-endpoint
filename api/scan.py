from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import Response, PlainTextResponse
import httpx, io, math
import numpy as np
import cv2 as cv
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Scan Endpoint", version="1.0")

# ---------- kleine Hilfen ----------

def url_to_image_bytes(url: str) -> bytes:
    timeout = httpx.Timeout(30.0, connect=10.0)
    headers = {"User-Agent": "scan-endpoint/1.0"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        r = client.get(url)
        if r.status_code != 200 or not r.content:
            raise HTTPException(502, detail=f"Fehler beim Laden: HTTP {r.status_code}")
        return r.content

def imread_from_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, detail="Bild konnte nicht dekodiert werden")
    return img

def to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))

def to_bgr(img_pil: Image.Image) -> np.ndarray:
    return cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)

# ---------- Vorverarbeitung / „Scanlook“ ----------

def white_balance_grayworld(img: np.ndarray) -> np.ndarray:
    # robust & billig
    b, g, r = cv.split(img.astype(np.float32))
    mb, mg, mr = [x.mean() for x in (b, g, r)]
    k = (mb + mg + mr) / 3.0 + 1e-6
    b *= k / mb; g *= k / mg; r *= k / mr
    out = cv.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def soft_binarize(gray: np.ndarray) -> np.ndarray:
    # adaptive, aber „soft“ gemischt mit Graustufen – verhindert harte Ausfransungen
    # 1) lokale Kontrastverstärkung
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    # 2) leichte Glättung
    g2 = cv.bilateralFilter(g1, d=5, sigmaColor=25, sigmaSpace=25)
    # 3) adaptive Schwelle
    th = cv.adaptiveThreshold(g2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, 31, 10)
    # 4) „weiche“ Mischung (0.65 binär / 0.35 grau)
    mixed = (0.65 * th + 0.35 * g2).astype(np.uint8)
    return mixed

def unsharp_mask(pil_img: Image.Image, radius=1.2, amount=1.4) -> Image.Image:
    # klassisches Unsharp – gute Lesbarkeit bei Text
    return pil_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount*100), threshold=2))

# ---------- Belegsuche (mit QR-Ignorierung) ----------

def find_receipt_quad(img: np.ndarray, min_conf=0.7):
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5,5), 0)

    edges = cv.Canny(gray, 50, 150)
    edges = cv.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0

    for c in contours:
        area = cv.contourArea(c)
        if area < (w*h)*0.03:  # zu klein
            continue
        # QR-ähnliche Quadrate rausfiltern
        rect = cv.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect
        ar = min(rw, rh) / (max(rw, rh) + 1e-6)
        if 0.80 < ar < 1.20 and area < (w*h)*0.25:
            # ziemliches Quadrat und nicht sehr groß -> sehr wahrscheinlich QR/Sticker
            continue

        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv.isContourConvex(approx):
            # Score: Größe * Seitenverhältnis „Beleg-typisch“ (sehr hochkant)
            poly = approx.reshape(4,2).astype(np.float32)
            rect2 = cv.minAreaRect(approx)
            (rw, rh) = rect2[1]
            tall = max(rw, rh) / (min(rw, rh) + 1e-6)  # >=1
            score = (area/(w*h)) * (min(tall/2.5, 1.0))  # tall ~ 2.5 begünstigen
            if score > best_score:
                best_score = score
                best = poly

    conf = float(min(best_score*1.4, 1.0))  # grobe Normierung
    return best, conf

def four_point_warp(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # sort points
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl

    (tl, tr, br, bl) = rect
    wA = np.linalg.norm(br-bl); wB = np.linalg.norm(tr-tl)
    hA = np.linalg.norm(tr-br); hB = np.linalg.norm(tl-bl)
    maxW = int(max(wA, wB)); maxH = int(max(hA, hB))
    maxW = max(200, maxW); maxH = max(200, maxH)

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(img, M, (maxW, maxH), flags=cv.INTER_CUBIC)
    return warped

# ---------- Upscaling ----------

def maybe_upscale_for_text(img_bgr: np.ndarray, mode: str = "auto") -> np.ndarray:
    if mode == "none":
        return img_bgr

    h, w = img_bgr.shape[:2]
    shorter = min(h, w)
    # heuristik: unter 900 px -> hochskalieren
    if mode in ("auto", "fast") and shorter < 900:
        scale = 2.0 if shorter < 700 else 1.5
        new_size = (int(w*scale), int(h*scale))
        up = cv.resize(img_bgr, new_size, interpolation=cv.INTER_LANCZOS4)
        return up

    if mode == "sr":
        # Optional: DNN Super-Resolution, falls Modelle vorhanden
        try:
            from cv2.dnn_superres import DnnSuperResImpl_create
            sr = DnnSuperResImpl_create()
            # Beispiel: ESPCN x2 (schnell), lade modelldatei aus /app/models/
            sr.readModel("/app/models/ESPCN_x2.pb")
            sr.setModel("espcn", 2)
            up = sr.upsample(img_bgr)
            return up
        except Exception:
            # Fallback
            return maybe_upscale_for_text(img_bgr, "fast")

    return img_bgr

# ---------- PDF-Erzeugung mit Größenlimit ----------

def make_pdf_bytes(img_bgr: np.ndarray, target_kb: int = 600, hard_cap_kb: int = 800) -> bytes:
    # Wir betten ein JPEG (Graustufen) in die PDF & schrauben Qualität bis unter Zielgröße
    # 1) nach PIL & Graustufen
    pil = to_pil(img_bgr).convert("L")

    def jpeg_bytes(quality: int) -> bytes:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        return buf.getvalue()

    # Quality-Schleife: Start 75 → runter
    q = 75
    jpg = jpeg_bytes(q)
    while len(jpg) > target_kb*1024 and q > 35:
        q -= 5
        jpg = jpeg_bytes(q)

    # notfalls hard cap
    if len(jpg) > hard_cap_kb*1024:
        while len(jpg) > hard_cap_kb*1024 and q > 25:
            q -= 3
            jpg = jpeg_bytes(q)

    # 2) PDF
    img_reader = ImageReader(io.BytesIO(jpg))
    # setze Seitenmaß so, dass DPI um 150–170 liegt
    wpx, hpx = pil.size
    dpi = 160.0
    pw, ph = (wpx / dpi * 72.0, hpx / dpi * 72.0)  # PDF-Punkte
    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=(pw, ph))
    c.drawImage(img_reader, 0, 0, width=pw, height=ph, preserveAspectRatio=True, mask='auto')
    c.showPage(); c.save()
    return pdf_buf.getvalue()

# ---------- Hauptpipeline ----------

def process_pipeline(img_bgr: np.ndarray,
                     crop_conf: float = 0.7,
                     upscale_mode: str = "auto",
                     bw_strength: float = 0.8) -> np.ndarray:
    # 0) Weißabgleich + milde Denoise
    img = white_balance_grayworld(img_bgr)
    img = cv.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)

    # 1) evtl. Upscaling
    img = maybe_upscale_for_text(img, mode=upscale_mode)

    # 2) sichere Belegsuche
    quad, conf = find_receipt_quad(img, min_conf=crop_conf)
    if quad is not None and conf >= crop_conf:
        img = four_point_warp(img, quad)

    # 3) „Scanlook“ (grau/bw gemischt) + sanfte Schärfung
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bw = soft_binarize(gray)

    # Mischung steuern
    bw = (bw_strength * bw + (1.0 - bw_strength) * gray).astype(np.uint8)

    pil = Image.fromarray(bw)
    pil = unsharp_mask(pil, radius=1.0, amount=1.1)  # moderat
    out = cv.cvtColor(np.array(pil), cv.COLOR_GRAY2BGR)
    return out

# ---------- Routes ----------

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.get("/scan")
def scan(
    file_url: str = Query(..., description="Direkter Bild-URL (JPG/PNG)"),
    max_kb: int = Query(600, ge=60, le=2000, description="Zielgröße PDF in KB (soft)"),
    hard_cap_kb: int = Query(800, ge=80, le=4000, description="harte Obergrenze PDF in KB"),
    crop_conf: float = Query(0.70, ge=0.0, le=1.0, description="Sicherheit für Zuschneiden/Begradigen"),
    upscale: str = Query("auto", regex="^(auto|fast|sr|none)$", description="Upscaling-Modus"),
    bw_strength: float = Query(0.80, ge=0.4, le=1.0, description="0.4=graulastig … 1.0=sehr hart schwarz/weiß"),
    filename: str = Query("scan.pdf")
):
    """
    Lädt ein Bild, macht daraus einen gut lesbaren „Scan“-PDF
    - Zuschneiden/Begradigen nur bei sicherer Erkennung (crop_conf)
    - QR-Codes werden bei der Kontursuche ignoriert
    - Optionales Upscaling für kleine/quellige Fotos
    - PDF-Größe wird automatisch unter das Ziel gedrückt
    """
    raw = url_to_image_bytes(file_url)
    img = imread_from_bytes(raw)

    out = process_pipeline(
        img_bgr=img,
        crop_conf=crop_conf,
        upscale_mode=upscale,
        bw_strength=bw_strength
    )

    pdf_bytes = make_pdf_bytes(out, target_kb=max_kb, hard_cap_kb=hard_cap_kb)

    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "Cache-Control": "no-store"
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
