# api/scan.py
from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
import httpx, io, math, time
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Scan Endpoint", version="1.0")

# ---------- Helpers ----------

async def fetch_bytes(url: str, timeout=30) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download fehlgeschlagen: {e}")

def pil_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    # EXIF-Orientation respektieren
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im.convert("RGB")

def _projection_score(img_arr: np.ndarray, angle_deg: float) -> float:
    # kleines, schnelles Schräglagen-Maß: Summenvarianz projizierter Zeilen
    angle = np.deg2rad(angle_deg)
    h, w = img_arr.shape
    # Rotations-Matrix (um Mittelpunkt)
    cos, sin = np.cos(angle), np.sin(angle)
    cx, cy = w/2, h/2
    # Stichprobe: nur jedes 2. Pixel (Speed)
    ys = np.arange(0, h, 2)
    xs = np.arange(0, w, 2)
    grid_x, grid_y = np.meshgrid(xs, ys)
    x = (grid_x - cx) * cos + (grid_y - cy) * sin + cx
    y = -(grid_x - cx) * sin + (grid_y - cy) * cos + cy
    # gültige Pixel
    mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    vals = img_arr[grid_y[mask], grid_x[mask]]
    # auf gerundete Zeilen projizieren
    rows = np.clip(np.rint(y[mask]).astype(int), 0, h-1)
    sums = np.bincount(rows, weights=vals, minlength=h)
    return float(np.var(sums))

def auto_deskew(gray: Image.Image) -> Image.Image:
    # für Speed verkleinern
    small = gray.resize((max(1, gray.width//2), max(1, gray.height//2)), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    # leichte Kantenverstärkung
    arr = arr - arr.mean()
    # Winkel scannen
    best_angle, best_score = 0.0, -1.0
    for ang in np.linspace(-10.0, 10.0, 41):  # Schritt 0.5°
        score = _projection_score((255 - arr).astype(np.uint8), ang)
        if score > best_score:
            best_score, best_angle = score, ang
    if abs(best_angle) < 0.2:
        return gray
    return gray.rotate(best_angle, expand=True, resample=Image.BICUBIC, fillcolor=255)

def preprocess(img: Image.Image) -> Image.Image:
    # 1) sanft entrauschen
    work = img.filter(ImageFilter.MedianFilter(size=3))
    # 2) in Grau
    gray = ImageOps.grayscale(work)
    # 3) automatische Schräglage
    gray = auto_deskew(gray)
    # 4) adaptive Binarisierung (sanft)
    arr = np.asarray(gray, dtype=np.uint8)
    # adaptive mean threshold
    k = 15  # Fenster
    pad = k // 2
    pad_arr = np.pad(arr, pad, mode='reflect')
    cumsum = pad_arr.cumsum(axis=0).cumsum(axis=1)
    H, W = arr.shape
    means = (
        cumsum[k:, k:] - cumsum[:-k, k:] - cumsum[k:, :-k] + cumsum[:-k, :-k]
    ) / (k * k)
    means = means.astype(np.float32)
    thresh = (means - 6).clip(0, 255)  # kleiner Offset
    bw = (arr[pad:pad+H, pad:pad+W] < thresh).astype(np.uint8) * 255
    # 5) leichte Unschärfe + Schärfen gegen Treppchen
    bw_img = Image.fromarray(bw, mode="L").filter(ImageFilter.GaussianBlur(0.4))
    bw_img = ImageOps.autocontrast(bw_img)
    return bw_img

def make_pdf_from_bw(bw_img: Image.Image, page_size=A4, margin_mm=8) -> bytes:
    # schwarze Pixel → Text, weißer Hintergrund
    if bw_img.mode != "L":
        bw_img = bw_img.convert("L")

    # Seite & Ränder
    page_w, page_h = page_size
    margin = margin_mm * mm
    max_w = page_w - 2 * margin
    max_h = page_h - 2 * margin

    # Bildseitenverhältnis
    img_w, img_h = bw_img.size
    scale = min(max_w / img_w, max_h / img_h)
    out_w, out_h = img_w * scale, img_h * scale
    x = (page_w - out_w) / 2
    y = (page_h - out_h) / 2

    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=page_size)

    # Wichtig: ImageReader statt BytesIO → Fix deines Fehlers
    png_buf = io.BytesIO()
    bw_img.save(png_buf, format="PNG", optimize=True)
    png_buf.seek(0)
    img_reader = ImageReader(png_buf)

    c.drawImage(img_reader, x, y, width=out_w, height=out_h,
                preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    bio.seek(0)
    return bio.read()

async def upload_tmpfiles(pdf_bytes: bytes, filename: str) -> str | None:
    # tmpfiles.org: multipart/form-data
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            files = {"file": (filename, pdf_bytes, "application/pdf")}
            r = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
            r.raise_for_status()
            data = r.json()
            # API liefert z.B. {"status": "ok", "data": {"url": "https://tmpfiles.org/dl/<id>/filename.pdf"}}
            url = data.get("data", {}).get("url")
            return url
    except Exception:
        return None

async def upload_0x0(pdf_bytes: bytes, filename: str) -> str | None:
    # 0x0.st akzeptiert application/octet-stream (plain Body mit ?file=@)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            files = {"file": (filename, pdf_bytes, "application/pdf")}
            r = await client.post("https://0x0.st", files=files)
            r.raise_for_status()
            txt = r.text.strip()
            # gibt direkte URL zurück (z.B. https://0x0.st/abcd.pdf)
            if txt.startswith("http"):
                return txt
    except Exception:
        pass
    return None

async def upload_with_fallbacks(pdf_bytes: bytes, filename: str) -> str:
    for uploader in (upload_tmpfiles, upload_0x0):
        url = await uploader(pdf_bytes, filename)
        if url:
            return url
    raise HTTPException(status_code=502, detail="Upload fehlgeschlagen (alle Hosts).")

# ---------- Routes ----------

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK. Nutze /scan?file_url=...&response=link|pdf"

@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Öffentliche URL zur Bilddatei"),
    response: str = Query("link", pattern="^(link|pdf)$", description="Antwortformat: link (default) oder pdf"),
    page: str = Query("A4", pattern="^(A4|letter)$", description="PDF-Seite")
):
    # 1) Download
    raw = await fetch_bytes(file_url)
    # 2) Bild laden & vorbereiten
    img = pil_from_bytes(raw)
    # Wenn das Foto quer liegt (breiter als hoch), drehe in Hochformat,
    # aber nur wenn es deutlich quer ist
    if img.width > img.height * 1.1:
        img = img.rotate(90, expand=True, resample=Image.BICUBIC)

    bw = preprocess(img)

    # 3) PDF bauen
    page_size = A4 if page == "A4" else letter
    pdf_bytes = make_pdf_from_bw(bw, page_size=page_size)

    if response == "pdf":
        # Direktes PDF zurück (praktisch zum Testen im Browser)
        return StreamingResponse(io.BytesIO(pdf_bytes),
                                 media_type="application/pdf",
                                 headers={"Content-Disposition": 'inline; filename="scan.pdf"'})
    else:
        # 4) Upload & Link zurückgeben
        filename = f"scan_{int(time.time())}.pdf"
        link = await upload_with_fallbacks(pdf_bytes, filename)
        return JSONResponse({"link": link, "filename": filename, "content_type": "application/pdf"})
