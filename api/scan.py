# scan.py
import io
import os
import base64
import urllib.parse
import httpx
from PIL import Image, ImageOps
import numpy as np
import cv2  # opencv-python-headless
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

app = FastAPI(title="Scan Endpoint")


async def download_to_bytes(url: str) -> bytes:
    timeout = httpx.Timeout(30.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code >= 400:
            raise HTTPException(400, f"Could not download file_url (HTTP {r.status_code})")
        return r.content


def deskew_and_crop(img_cv: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
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
                best = approx.reshape(4, 2).astype(np.float32)
                best_area = area

    if best is not None and best_area > 0.1 * (w * h):
        def order(pts):
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            return np.array([tl, tr, br, bl], dtype=np.float32)

        quad = order(best)
        widthA = np.linalg.norm(quad[2] - quad[3])
        widthB = np.linalg.norm(quad[1] - quad[0])
        heightA = np.linalg.norm(quad[1] - quad[2])
        heightB = np.linalg.norm(quad[0] - quad[3])
        maxW = int(max(widthA, widthB))
        maxH = int(max(heightA, heightB))
        dest = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(quad, dest)
        warped = cv2.warpPerspective(img_cv, M, (maxW, maxH))
    else:
        warped = img_cv

    gray2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 15
    )
    return thr


def image_bytes_to_pdf_bytes(img_np: np.ndarray) -> bytes:
    pil_img = Image.fromarray(img_np)
    pil_img = ImageOps.invert(ImageOps.invert(pil_img))
    buf = io.BytesIO()
    pil_img.convert("L").save(buf, format="PDF", resolution=300.0)
    return buf.getvalue()


async def upload_temp_file(pdf_bytes: bytes, filename: str) -> str:
    """Upload to transfer.sh and return public URL"""
    async with httpx.AsyncClient() as client:
        files = {filename: io.BytesIO(pdf_bytes)}
        r = await client.post(f"https://transfer.sh/{filename}", content=pdf_bytes)
        if r.status_code >= 400:
            raise HTTPException(500, f"Upload failed ({r.status_code})")
        return r.text.strip()


@app.post("/scan")
async def scan(
    file_url: str = Query(...),
):
    if not file_url:
        raise HTTPException(422, "file_url is required")

    raw = await download_to_bytes(file_url)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    scanned = deskew_and_crop(cv_img)
    pdf_bytes = image_bytes_to_pdf_bytes(scanned)

    filename = "scan.pdf"
    public_url = await upload_temp_file(pdf_bytes, filename)

    return JSONResponse({
        "filename": filename,
        "url": public_url
    })


@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"
