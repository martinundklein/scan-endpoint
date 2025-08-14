# api/scan.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
import io, math, asyncio
import httpx
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2 as cv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Light Scan API")

# ---------- Hilfsfunktionen ----------

async def fetch_bytes(url: str, timeout: float = 15.0) -> bytes:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code != 200 or not r.content:
            raise HTTPException(400, detail=f"Download failed (status={r.status_code})")
        return r.content

def pil_from_any(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    # Manche JPEGs haben EXIF-Orientation -> korrigieren
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img

def to_cv_gray(pil_im: Image.Image) -> np.ndarray:
    if pil_im.mode == "L":
        arr = np.array(pil_im)
    else:
        arr = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2GRAY)
    return arr

def gentle_deskew(gray: np.ndarray, max_angle_deg: float = 20.0) -> np.ndarray:
    """
    Deskew NUR wenn plausibel:
    - Kanten maske → minAreaRect Winkel
    - nur leichte Rotation (|theta| <= max_angle_deg)
    - kein Perspective-Warp, nur rotate
    """
    # leichte Glättung + Kanten
    g = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(g, 50, 150)
    # genügend Kanten?
    if edges.mean() < 5:   # sehr „ruhiges“ Bild -> keine Aussage
        return gray
    # Konturen und größte zusammenhängende Maske
    cnts, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray
    cnt = max(cnts, key=cv.contourArea)
    area = cv.contourArea(cnt)
    h, w = gray.shape[:2]
    if area < 0.10 * (w * h):  # zu klein -> unsicher
        return gray
    rect = cv.minAreaRect(cnt)  # ((cx,cy),(rw,rh),angle)
    angle = rect[-1]
    # OpenCV-MinAreaRect Winkel: [-90,0)
    if angle < -45:
        angle += 90.0
    if abs(angle) > max_angle_deg:
        return gray  # unsicher -> nicht drehen

    # Rotieren um Bildmitte
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv.warpAffine(gray, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return rotated

def bw_scan_look(gray: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Sanfter „Scan“-Look:
    - Kontraststreckung
    - adaptive Schwelle + Mischung mit Graubild (vermeidet harte Posterization)
    - QR/feines Moiré dämpfen (kleine Median-Entspr.)
    """
    # Kontrast strecken
    g = cv.normalize(gray, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # leichte Schärfung
    g_blur = cv.GaussianBlur(g, (0, 0), 1.0)
    g_sharp = cv.addWeighted(g, 1.25, g_blur, -0.25, 0)

    # adaptive Threshold (robust bei ungleichmäßiger Beleuchtung)
    thr = cv.adaptiveThreshold(
        g_sharp, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 31, 10
    )

    # Mischung: (1-strength) * grau + strength * binär
    out = cv.addWeighted(g_sharp, (1.0 - strength), thr, strength, 0)
    # leichte Entspr. gegen starkes Rauschen
    out = cv.medianBlur(out, 3)
    return out

def jpeg_bytes_from_array(arr: np.ndarray, quality: int = 85) -> bytes:
    pil = Image.fromarray(arr if arr.ndim == 2 else cv.cvtColor(arr, cv.COLOR_BGR2RGB))
    if pil.mode != "L":
        pil = pil.convert("L")  # Graustufen spart Größe ohne Lesbarkeit zu verlieren
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()

def build_pdf_from_jpeg(jpeg_data: bytes, img_w: int, img_h: int) -> bytes:
    """
    Erzeugt ein 1‑seitiges PDF, Seite an Bildseitenverhältnis angepasst (max A4).
    """
    # Seite: maximal A4, aber Verhältnis vom Bild beibehalten
    max_w, max_h = A4  # points (1/72 inch)
    # DPI fürs Layout (wir nehmen 150dpi als solide Basis)
    dpi = 150.0
    # Zielgröße in Points
    page_w = min(max_w, img_w / dpi * 72.0)
    page_h = min(max_h, img_h / dpi * 72.0)
    # Verhältnis checken
    img_ratio = img_w / max(1.0, img_h)
    page_ratio = page_w / max(1.0, page_h)
    if img_ratio > page_ratio:
        page_h = page_w / img_ratio
    else:
        page_w = page_h * img_ratio

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))
    img = ImageReader(io.BytesIO(jpeg_data))
    c.drawImage(img, 0, 0, width=page_w, height=page_h, preserveAspectRatio=True, mask=None)
    c.showPage()
    c.save()
    return buf.getvalue()

def fit_under_kb(arr: np.ndarray, max_kb: int, hard_cap_kb: int) -> bytes:
    """
    Iteriert Qualität + Skalierung bis PDF < max_kb (bzw. < hard_cap_kb als letzte Stufe).
    Ziel: schnell (kleine Schleife), nicht perfekt.
    """
    h, w = arr.shape[:2]
    # Startwerte
    scales = [1.0, 0.9, 0.8, 0.7, 0.6]
    qualities = [85, 80, 75, 70, 65, 60, 55, 50, 45, 40]

    best = None
    for s in scales:
        new_w = max(600, int(w * s))
        new_h = max(600, int(h * s * (h / max(1, h))))  # proportional
        resized = cv.resize(arr, (new_w, int(new_h)), interpolation=cv.INTER_AREA)
        for q in qualities:
            jpg = jpeg_bytes_from_array(resized, quality=q)
            pdf = build_pdf_from_jpeg(jpg, new_w, resized.shape[0])
            kb = len(pdf) // 1024
            if best is None or kb < len(best) // 1024:
                best = pdf
            if kb <= max_kb:
                return pdf
            # wenn knapp über max, aber unter hart -> merken
            if kb <= hard_cap_kb and best is not None:
                best = pdf
        # nächste Skalierung
    # Fallback: bestes bisher (unter harten Cap wenn möglich)
    if best is None:
        raise HTTPException(500, detail="Failed to compress PDF")
    if len(best) // 1024 > hard_cap_kb:
        # letzte Notbremse: brutales Downscale
        tiny = cv.resize(arr, (min(900, arr.shape[1] // 2), min(1400, arr.shape[0] // 2)), interpolation=cv.INTER_AREA)
        pdf = build_pdf_from_jpeg(jpeg_bytes_from_array(tiny, 45), tiny.shape[1], tiny.shape[0])
        return pdf
    return best

# ---------- API ----------

@app.get("/", response_class=PlainTextResponse)
def root():
    return "Light Scan API OK"

@app.get("/scan")
async def scan(
    file_url: str = Query(..., description="Direkter Bild-URL (JPG/PNG)"),
    filename: str = Query("scan.pdf"),
    max_kb: int = Query(600, ge=50, le=4096, description="Zielgröße in KB"),
    hard_cap_kb: int = Query(800, ge=50, le=8192, description="Absolute Obergrenze in KB"),
    bw_strength: float = Query(0.8, ge=0.0, le=1.0, description="0=grau, 1=hartes Schwarz/Weiß"),
    deskew: bool = Query(True, description="Leichtes Begradigen, nur wenn plausibel"),
    max_angle: float = Query(20.0, ge=0.0, le=45.0, description="Maximaler Deskew-Winkel")
):
    # 1) Download
    data = await fetch_bytes(file_url)

    # 2) PIL laden
    pil = pil_from_any(data)
    # weicher Downscale für sehr große Bilder (Zeit & Größe)
    MAX_SIDE = 2200  # schnell + ausreichend für Belege
    if max(pil.size) > MAX_SIDE:
        pil = ImageOps.contain(pil, (MAX_SIDE, MAX_SIDE), method=Image.LANCZOS)

    # 3) nach Grau + optional Deskew
    gray = to_cv_gray(pil)
    if deskew:
        gray = gentle_deskew(gray, max_angle_deg=max_angle)

    # 4) S/W‑Scan‑Look
    bw = bw_scan_look(gray, strength=bw_strength)

    # 5) Packen → PDF unter Zielgröße
    pdf_bytes = fit_under_kb(bw, max_kb=max_kb, hard_cap_kb=hard_cap_kb)

    # 6) Response
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"'
        }
    )
