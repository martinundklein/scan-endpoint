from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, JSONResponse
import asyncio
import httpx
import io
import math
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, portrait
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Receipt Scan", version="1.0.0")

# --------- Helfer ---------

async def fetch_image(url: str, timeout: float = 20.0) -> Image.Image:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(400, f"file_url fetch failed: {r.status_code}")
        data = r.content
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"cannot open image: {e}")
    return img


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def resize_max(img: np.ndarray, max_side: int = 1600) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def order_quad(pts: np.ndarray) -> np.ndarray:
    # ordnet 4 Punkte für perspectiveTransform
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_quad(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    maxWidth = max(maxWidth, 100)
    maxHeight = max(maxHeight, 100)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    return warped


def score_receipt_contour(cnt: np.ndarray, img_area: float) -> float:
    # Score: groß, länglich (kein Quadrat), Vierpunkte-Approx gut
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    # reject keine 4 Punkte
    if len(approx) != 4:
        return -1.0

    x, y, w, h = cv2.boundingRect(approx)
    if w <= 0 or h <= 0:
        return -1.0

    aspect = max(w, h) / (min(w, h) + 1e-6)
    # QR‑Codes sind meist ~quadratisch => raus, wenn zu quadratisch
    if 0.85 <= (w / (h + 1e-6)) <= 1.15:
        return -1.0

    area = cv2.contourArea(approx)
    area_ratio = float(area) / float(img_area)

    # Längliches Verhältnis bevorteilen
    elong = min(aspect / 2.0, 1.0)  # cap bei 1
    # Score kombinieren
    score = 0.65 * area_ratio + 0.35 * elong
    return score


def try_detect_receipt(image: np.ndarray, conf_threshold: float = 0.70) -> Tuple[Optional[np.ndarray], float]:
    work, scale = resize_max(image, 1400)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # leichte Glättung gegen Teppichrauschen
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # Kanten
    edges = cv2.Canny(gray, 40, 120)
    # schließen, um Lücken zu reduzieren
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = work.shape[0] * work.shape[1]

    best = None
    best_score = -1.0
    for c in cnts:
        score = score_receipt_contour(c, img_area)
        if score > best_score:
            best_score = score
            best = c

    if best is None or best_score < conf_threshold:
        return None, best_score

    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    approx = approx.reshape(-1, 2).astype("float32")

    warped = four_point_transform(work, approx)
    # zurück zur ursprünglichen Auflösung hochskalieren
    if scale != 1.0:
        inv = 1.0 / scale
        warped = cv2.resize(
            warped,
            (int(warped.shape[1] * inv), int(warped.shape[0] * inv)),
            interpolation=cv2.INTER_CUBIC,
        )
    return warped, best_score


def to_soft_bw(image: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Sanftes S/W: (CLAHE + adaptives Threshold) mit Blend."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Kontrast lokal anheben
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # adaptives Threshold (nicht zu hart)
    thr = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    # In ein 3‑Kanal-Bild
    thr_color = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
    gray_color = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    # sanft blenden (strength steuert "Scannigkeit")
    out = cv2.addWeighted(gray_color, 1.0 - strength, thr_color, strength, 0)
    return out


def ensure_portrait(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    # Belege sind fast immer hochkant
    if w > h:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def image_to_pdf_bytes(img_bgr: np.ndarray) -> bytes:
    img_bgr = ensure_portrait(img_bgr)

    # als PNG in RAM
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    png_bytes = io.BytesIO(buf.tobytes())

    # PDF bauen
    packet = io.BytesIO()
    # Seite: A4 Portrait
    c = canvas.Canvas(packet, pagesize=portrait(A4))
    page_w, page_h = portrait(A4)

    # Bildabmessungen
    pil_img = Image.open(io.BytesIO(png_bytes.getvalue()))
    iw, ih = pil_img.size
    aspect = iw / ih

    # mit Rändern
    margin = 24  # pt
    max_w = page_w - 2 * margin
    max_h = page_h - 2 * margin

    # skaliere so, dass alles drauf passt
    if max_w / max_h < aspect:
        out_w = max_w
        out_h = out_w / aspect
    else:
        out_h = max_h
        out_w = out_h * aspect

    x = (page_w - out_w) / 2
    y = (page_h - out_h) / 2

    c.drawImage(ImageReader(io.BytesIO(png_bytes.getvalue())),
                x, y, width=out_w, height=out_h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    return packet.getvalue()


async def upload_fallback(pdf_bytes: bytes, filename: str = "scan.pdf") -> str:
    """
    Nur falls du ?response=link nutzt. Probiert file.io, dann tmpfiles.org.
    """
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        # 1) file.io
        try:
            files = {"file": (filename, pdf_bytes, "application/pdf")}
            r = await client.post("https://file.io", files=files)
            if r.status_code == 200:
                j = r.json()
                if j.get("success") and j.get("link"):
                    return j["link"]
        except Exception:
            pass

        # 2) tmpfiles.org API
        try:
            files = {"file": (filename, pdf_bytes, "application/pdf")}
            r = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
            if r.status_code == 200:
                j = r.json()
                link = j.get("data", {}).get("url")
                if link:
                    # Direktlink zur Datei (nicht nur Seite)
                    # tmpfiles liefert Seite + Pfad; der direkte Link ist oft identisch
                    return link
        except Exception:
            pass

    raise HTTPException(502, "upload failed")


# --------- API ---------

@app.get("/", tags=["meta"])
async def root():
    return {"ok": True, "service": "receipt-scan", "version": "1.0.0"}


@app.get("/scan", tags=["scan"])
async def scan(
    file_url: str = Query(..., description="Öffentlich abrufbare Bild-URL (Airtable Attachment URL)"),
    response: str = Query("pdf", pattern="^(pdf|link)$", description="pdf=PDF direkt, link=Upload-Link"),
    conf: float = Query(0.70, ge=0.0, le=0.99, description="Schwellenwert für Zuschneiden/Entzerren"),
    strength: float = Query(0.35, ge=0.15, le=0.6, description="Intensität der S/W-Optik")
):
    """
    Holt ein Foto, versucht den Beleg zu erkennen (QR-Codes werden verworfen),
    entzerrt nur bei ausreichender Sicherheit (>= conf),
    sonst nur S/W-Optik. Gibt standardmäßig ein PDF zurück.
    """
    pil_img = await fetch_image(file_url)
    bgr = pil_to_cv(pil_img)

    # 1) Belegsuche
    warped, score = try_detect_receipt(bgr, conf_threshold=conf)

    if warped is None:
        work = bgr.copy()  # kein sicherer Beleg -> nicht beschneiden
    else:
        work = warped

    # 2) sanftes S/W (weniger hart)
    processed = to_soft_bw(work, strength=strength)

    # 3) PDF
    try:
        pdf_bytes = image_to_pdf_bytes(processed)
    except Exception as e:
        raise HTTPException(500, f"pdf build failed: {e}")

    if response == "link":
        url = await upload_fallback(pdf_bytes, "scan.pdf")
        return JSONResponse({"url": url, "crop_confidence": round(max(score, 0.0), 3)})
    else:
        headers = {
            "Content-Disposition": 'inline; filename="scan.pdf"'
        }
        return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
