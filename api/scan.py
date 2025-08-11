from fastapi import FastAPI, Request, Response
import requests, io, cv2, numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

app = FastAPI()

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    dst = np.array([[0, 0],[maxW - 1, 0],[maxW - 1, maxH - 1],[0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped

def to_scanned_bw(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    return bw

def detect_document(image):
    ratio = image.shape[0] / 700.0
    small = cv2.resize(image, (int(image.shape[1]/ratio), 700))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype("float32") * ratio
            return four_point_transform(image, pts)
    return image

def pdf_bytes_from_bw(bw_np):
    import io
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    h, w = bw_np.shape[:2]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    pil = Image.fromarray(bw_np).convert("L").convert("RGB")
    img = ImageReader(pil)
    page_w, page_h = A4
    margin = 24
    scale = min((page_w-2*margin)/w, (page_h-2*margin)/h)
    draw_w, draw_h = w*scale, h*scale
    x = (page_w - draw_w)/2
    y = (page_h - draw_h)/2
    c.drawImage(img, x, y, draw_w, draw_h)
    c.showPage()
    c.save()
    return buf.getvalue()

@app.post("/scan")
async def scan(req: Request):
    data = await req.json()
    url = data.get("file_url")
    if not url:
        return Response("Missing file_url", status_code=400)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img_bytes = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return Response("Unsupported image", status_code=415)
    doc = detect_document(img)
    bw = to_scanned_bw(doc)
    pdf = pdf_bytes_from_bw(bw)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": "inline; filename=scan.pdf"})
