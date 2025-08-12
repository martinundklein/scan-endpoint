# api/scan.py
import io
import math
import os
import uuid
import asyncio
from typing import Literal, Tuple, Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.utils import ImageReader

APP = FastAPI(title="Scan Endpoint", version="1.0")

# -------------- Networking helpers --------------

TIMEOUT = httpx.Timeout(30.0, connect=15.0)
HEADERS = {"User-Agent": "scan-endpoint/1.0"}

async def fetch_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code >= 400:
            raise HTTPException(status_code=400, detail=f"Download failed ({r.status_code})")
        return r.content

async def upload_tmpfiles(data: bytes, filename: str) -> Optional[str]:
    # tmpfiles.org – simple POST multipart
    files = {"file": (filename, data, "application/pdf")}
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
        r = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
        if r.status_code == 200:
            try:
                j = r.json()
                # returns https://tmpfiles.org/dl/<id>/<name>
                return j.get("data", {}).get("url")
            except Exception:
                return None
    return None

async def upload_fileio(data: bytes, filename: str) -> Optional[str]:
    files = {"file": (filename, data, "application/pdf")}
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
        r = await client.post("https://file.io", files=files)
        if r.status_code == 200:
            try:
                j = r.json()
                if j.get("success"):
                    return j.get("link") or j.get("url")
            except Exception:
                return None
    return None

async def upload_transfersh(data: bytes, filename: str) -> Optional[str]:
    # Some regions block transfer.sh; keep as last fallback
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
        try:
            r = await client.put(f"https://transfer.sh/{filename}", content=data)
            if r.status_code in (200, 201):
                return r.text.strip()
        except Exception:
            return None
    return None

async def upload_with_fallbacks(pdf_bytes: bytes, filename: str) -> str:
    for fn in (upload_tmpfiles, upload_fileio, upload_transfersh):
        try:
            link = await fn(pdf_bytes, filename)
            if link:
                return link
        except Exception:
            continue
    raise HTTPException(status_code=502, detail="Upload failed on all providers")

# -------------- Image utils --------------

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def order_quad(pts: np.ndarray) -> np.ndarray:
    # pts shape (4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_quad(pts.astype(np.float32))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    maxW = max(maxW, 100)
    maxH = max(maxH, 100)
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_CUBIC)
    return warped

def largest_receipt_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    # emphasize long, thin rectangles
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0
    h, w = gray.shape[:2]
    img_area = h * w
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 0.02 * img_area:
            # favor tall aspect rectangles
            rect = cv2.minAreaRect(approx)
            (rw, rh) = rect[1]
            if rw == 0 or rh == 0:
                continue
            aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
            score = area * (1 + min(aspect, 6.0))
            if score > best_score:
                best_score = score
                best = approx.reshape(4,2)
    return best

def deskew_by_hough(gray: np.ndarray) -> np.ndarray:
    # estimate skew from text lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    angle = 0.0
    if lines is not None:
        angles = []
        for rho, theta in lines[:,0]:
            deg = (theta * 180.0 / np.pi) - 90.0
            if -45 < deg < 45:
                angles.append(deg)
        if len(angles) > 0:
            angle = float(np.median(angles))
    if abs(angle) > 0.2:
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return gray

def mask_qr(gray: np.ndarray) -> np.ndarray:
    # try QR detection and blur that region slightly so binarization ignores dense code
    detector = cv2.QRCodeDetector()
    retval, points = detector.detect(gray)
    mask = np.zeros_like(gray)
    if retval and points is not None:
        pts = points.astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        # expand
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT,(21,21)), iterations=1)
        # blur inside mask
        blurred = cv2.GaussianBlur(gray, (0,0), 3)
        gray = np.where(mask==255, blurred, gray).astype(np.uint8)
    return gray

def illumination_correct(gray: np.ndarray) -> np.ndarray:
    # White top-hat: removes slow-varying background
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (31,31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    # normalize
    return cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)

def sauvola_threshold(gray: np.ndarray) -> np.ndarray:
    # Sauvola via OpenCV ximgproc absent → emulate with mean/var in box filter
    gray_f = gray.astype(np.float32)
    ksize = 25
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(ksize, ksize))
    mean_sq = cv2.boxFilter(gray_f*gray_f, ddepth=-1, ksize=(ksize, ksize))
    var = mean_sq - mean*mean
    std = np.sqrt(np.maximum(var, 1e-6))
    R = 128.0
    k = 0.2
    thresh = mean * (1 + k * ((std / R) - 1))
    bw = (gray_f > thresh).astype(np.uint8) * 255
    # invert to black text on white
    bw = 255 - bw
    # cleanup
    bw = cv2.medianBlur(bw, 3)
    return bw

def find_and_warp_receipt(bgr: np.ndarray) -> np.ndarray:
    # downscale for speed, keep ratio
    h, w = bgr.shape[:2]
    scale = 1200.0 / max(h, w)
    if scale < 1.0:
        bgr_small = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        bgr_small = bgr.copy()
        scale = 1.0
    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)

    cnt = largest_receipt_contour(gray)
    if cnt is not None:
        warped_small = four_point_warp(bgr_small, cnt)
        if scale != 1.0:
            inv = 1.0/scale
            warped = cv2.resize(warped_small, (int(warped_small.shape[1]*inv), int(warped_small.shape[0]*inv)),
                                interpolation=cv2.INTER_CUBIC)
        else:
            warped = warped_small
        return warped

    # Fallback: deskew whole image and crop center
    gray = deskew_by_hough(gray)
    bgr_small = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        inv = 1.0/scale
        fallback = cv2.resize(bgr_small, (int(bgr_small.shape[1]*inv), int(bgr_small.shape[0]*inv)),
                              interpolation=cv2.INTER_CUBIC)
    else:
        fallback = bgr_small
    return fallback

def preprocess_pipeline(bgr: np.ndarray) -> np.ndarray:
    warped = find_and_warp_receipt(bgr)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)
    gray = mask_qr(gray)
    gray = illumination_correct(gray)
    bw = sauvola_threshold(gray)

    # thin text a bit, remove small speckles
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)), iterations=1)
    return bw

# -------------- PDF creation --------------

def make_pdf(bw_img: np.ndarray, page: Literal["A4","Letter"]="A4") -> bytes:
    # pad to small border
    pad = 12
    bw_pad = cv2.copyMakeBorder(bw_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    # to PIL (ReportLab wants an ImageReader)
    pil = Image.fromarray(bw_pad)
    bio = io.BytesIO()
    pil.save(bio, format="PNG")
    bio.seek(0)
    img_reader = ImageReader(bio)

    if page.upper()=="A4":
        pw, ph = A4
    else:
        pw, ph = letter

    # scale to fit keeping aspect
    iw, ih = pil.size
    # 72 dpi points
    max_w = pw - 36  # 18pt margins
    max_h = ph - 36
    scale = min(max_w/iw, max_h/ih)
    out_w = iw*scale
    out_h = ih*scale
    x = (pw - out_w)/2
    y = (ph - out_h)/2

    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=(pw, ph))
    c.drawImage(img_reader, x, y, width=out_w, height=out_h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    out.seek(0)
    return out.read()

# -------------- Endpoint --------------

@APP.get("/", response_class=PlainTextResponse)
async def root():
    return "scan-endpoint OK"

@APP.get("/scan")
async def scan(
    file_url: str = Query(..., description="Direct URL to the image (PNG/JPG/PDF first page)"),
    response: Literal["link","pdf","json"] = Query("link")
):
    # 1) download
    raw = await fetch_bytes(file_url)

    # 2) load image (if PDF, take first page via PIL)
    try:
        img = Image.open(io.BytesIO(raw))
        if getattr(img, "is_animated", False):  # e.g., multi‑frame tiff
            img.seek(0)
        # If PDF: PIL loads only first page when registered – if it fails, raise
    except Exception:
        raise HTTPException(status_code=415, detail="Unsupported input. Provide PNG/JPG/PDF (first page).")

    bgr = pil_to_cv(img)

    # 3) process
    bw = preprocess_pipeline(bgr)

    # 4) make pdf
    pdf_bytes = make_pdf(bw, page="A4")
    fname = f"scan_{uuid.uuid4().hex[:8]}.pdf"

    if response == "pdf":
        return StreamingResponse(io.BytesIO(pdf_bytes),
                                 media_type="application/pdf",
                                 headers={"Content-Disposition": f'inline; filename="{fname}"'})
    elif response == "json":
        link = await upload_with_fallbacks(pdf_bytes, fname)
        return JSONResponse({"filename": fname, "url": link})
    else:  # link
        link = await upload_with_fallbacks(pdf_bytes, fname)
        return PlainTextResponse(link)
