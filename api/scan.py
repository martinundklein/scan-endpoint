import io
import os
import urllib.parse
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image, ImageOps
import numpy as np
import cv2  # opencv-python-headless

app = FastAPI(title="Scan-to-PDF Link", version="1.2.0")


# ---------- Helpers

async def fetch(url: str) -> tuple[bytes, Optional[str]]:
    """Lädt Bytes + Content-Type (falls vorhanden) mit Redirect-Follow."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code >= 400:
            raise HTTPException(400, f"Download fehlgeschlagen (HTTP {r.status_code})")
        return r.content, r.headers.get("content-type")


def guess_filename_from_url(url: str, default: str = "scan.pdf") -> str:
    name = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
    if not name:
        return default
    if not name.lower().endswith(".pdf"):
        name = f"{os.path.splitext(name)[0]}.pdf"
    return name


def perspective_warp_or_deskew(bgr: np.ndarray) -> np.ndarray:
    """Versucht Dokument als Viereck zu finden → Perspektivkorrektur.
    Fallback: nur leichte Rotation/Deskew durch Hough-Linien."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    best = None
    best_area = 0

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best = approx.reshape(4, 2).astype(np.float32)

    if best is not None and best_area > 0.15 * (w * h):
        # Ordnung: tl, tr, br, bl
        pts = best
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        quad = np.array([tl, tr, br, bl], dtype=np.float32)

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxW = int(max(widthA, widthB))
        maxH = int(max(heightA, heightB))
        maxW = max(300, min(maxW, 4000))
        maxH = max(300, min(maxH, 4000))

        dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(quad, dst)
        warped = cv2.warpPerspective(bgr, M, (maxW, maxH))
        return warped

    # Fallback: grobe Schräglagenkorrektur
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for rho, theta in lines[:, 0]:
            a = (theta * 180 / np.pi) % 180
            if 80 < a < 100:  # nahe an vertikal
                angles.append(a - 90)
        if angles:
            angle = np.median(angles)
    if abs(angle) > 0.5:
        center = (w // 2, h // 2)
        M2 = cv2.getRotationMatrix2D(center, angle, 1.0)
        bgr = cv2.warpAffine(bgr, M2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return bgr


def image_to_scanned_pdf(image_bytes: bytes) -> bytes:
    """Bild → perspektivkorrigiert → SW-Scan-Look → PDF (eine Seite)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    warped = perspective_warp_or_deskew(bgr)

    gray2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # adaptiver Threshold für „Scan“-Look
    scan = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 15
    )

    # kleine Rauschentfernung / Öffnung
    kernel = np.ones((2, 2), np.uint8)
    scan = cv2.morphologyEx(scan, cv2.MORPH_OPEN, kernel)

    pil = Image.fromarray(scan)
    buf = io.BytesIO()
    pil.convert("L").save(buf, format="PDF", resolution=300.0)
    return buf.getvalue()


async def upload_to_transfersh(pdf_bytes: bytes, filename: str) -> str:
    """Upload zu transfer.sh, Text-URL zurück."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"https://transfer.sh/{filename}", content=pdf_bytes)
        if r.status_code >= 400:
            raise HTTPException(502, f"Upload fehlgeschlagen (HTTP {r.status_code})")
        return r.text.strip()


# ---------- Endpoints

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"


@app.get("/scan")
async def scan(file_url: str = Query(..., description="Öffentliche Datei-URL (Airtable Attachment)")):
    if not file_url:
        raise HTTPException(422, "Parameter file_url fehlt")

    # Datei laden
    raw, ctype = await fetch(file_url)

    # Bild → „Scan“-PDF, PDF → Durchlauf
    is_pdf = False
    if ctype:
        is_pdf = "pdf" in ctype.lower()
    else:
        # Fallback über Dateiendung
        is_pdf = file_url.lower().endswith(".pdf")

    if is_pdf:
        pdf_bytes = raw  # PDF unverändert durchreichen
    else:
        try:
            pdf_bytes = image_to_scanned_pdf(raw)
        except Exception as e:
            raise HTTPException(400, f"Bildverarbeitung fehlgeschlagen: {e}")

    # Hochladen & Link zurück
    filename = guess_filename_from_url(file_url)
    link = await upload_to_transfersh(pdf_bytes, filename)
    return JSONResponse({"url": link})
