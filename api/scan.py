from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import io, base64, requests
import numpy as np
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Scan Endpoint")

class UrlBody(BaseModel):
    file_url: str | None = None

def read_image_from_upload(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=415, detail="Unsupported image")
    return img

def read_image_from_url(url: str) -> np.ndarray:
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Cannot fetch file_url: {e}")
    if r.status_code != 200 or not r.content:
        raise HTTPException(status_code=400, detail=f"Bad file_url (status {r.status_code})")
    # try image first
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # if PDF: take first page rasterization is out-of-scope; tell user to supply image
    raise HTTPException(status_code=415, detail="file_url is not an image")

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1]=pts[np.argmin(diff)]; rect[3]=pts[np.argmax(diff)]
    (tl,tr,br,bl) = rect
    widthA = np.linalg.norm(br-bl); widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br); heightB = np.linalg.norm(tl-bl)
    maxW = int(max(widthA,widthB)); maxH = int(max(heightA,heightB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))

def auto_scan(img: np.ndarray) -> np.ndarray:
    # resize for speed
    ratio = 1000.0 / max(img.shape[0], img.shape[1])
    small = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)

    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
    doc = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc = approx.reshape(4,2) / ratio
            break

    if doc is not None:
        scanned = four_point_transform(img, doc.astype("float32"))
    else:
        scanned = img  # fallback: no warp

    # grayscale + adaptive threshold for "scan look"
    g = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 25, 15)
    return thr

def image_to_pdf_bytes(bw_img: np.ndarray) -> bytes:
    # convert to PNG in-memory
    ok, png = cv2.imencode(".png", bw_img)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG encode failed")
    png_bytes = png.tobytes()

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    pil_img = ImageReader(io.BytesIO(png_bytes))
    iw, ih = pil_img.getSize()
    # fit to page (portrait A4)
    page_w, page_h = (595.2756, 841.8898)
    scale = min(page_w/iw, page_h/ih)
    w = iw*scale; h = ih*scale
    x = (page_w - w)/2; y = (page_h - h)/2
    c.setPageSize((page_w, page_h))
    c.drawImage(pil_img, x, y, w, h, preserveAspectRatio=True, mask='auto')
    c.showPage(); c.save()
    return buf.getvalue()

@app.post("/scan")
async def scan(
    response: str = Query("json", pattern="^(json|pdf)$"),
    file: UploadFile | None = File(default=None),
    url_body: UrlBody | None = None
):
    # 1) Eingabe holen (UploadFile ODER file_url)
    img = None
    if file is not None:
        img = read_image_from_upload(file)
    elif url_body and url_body.file_url:
        img = read_image_from_url(url_body.file_url)
    else:
        raise HTTPException(status_code=422, detail="Provide a file or file_url")

    # 2) verarbeiten
    bw = auto_scan(img)
    pdf_bytes = image_to_pdf_bytes(bw)

    if response == "pdf":
        return StreamingResponse(io.BytesIO(pdf_bytes),
                                 media_type="application/pdf",
                                 headers={"Content-Disposition": 'inline; filename="scan.pdf"'})
    # else json
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    return JSONResponse({
        "filename": "scan.pdf",
        "content_type": "application/pdf",
        "file_base64": b64
    })
