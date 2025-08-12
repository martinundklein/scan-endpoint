from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import requests
import cv2 as cv
import numpy as np
from io import BytesIO
from PIL import Image
import tempfile

app = FastAPI()

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    return img

def detect_document_edges(image, min_confidence=0.7):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150)

    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    for cnt in contours:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv.contourArea(approx)
            image_area = image.shape[0] * image.shape[1]
            confidence = area / image_area
            if confidence > min_confidence:
                return approx.reshape(4, 2)

    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def apply_scan_effect(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # adaptive threshold for "scan" look
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, 25, 15)
    # leicht weichzeichnen, um harte Kanten zu glätten
    bw = cv.medianBlur(bw, 3)
    return bw

@app.get("/scan")
def scan(file_url: str = Query(...)):
    img = download_image(file_url)

    # Versuche Begradigung
    pts = detect_document_edges(img, min_confidence=0.7)
    if pts is not None:
        img = four_point_transform(img, pts)

    # Scan-Look anwenden
    scanned = apply_scan_effect(img)

    # zurück als PDF
    pil_img = Image.fromarray(scanned)
    pdf_bytes = BytesIO()
    pil_img.save(pdf_bytes, format="PDF")
    pdf_bytes.seek(0)

    return StreamingResponse(pdf_bytes, media_type="application/pdf",
                              headers={"Content-Disposition": "inline; filename=scan.pdf"})
