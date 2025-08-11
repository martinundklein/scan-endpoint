# scan.py
import io, base64, json
from typing import Optional, Literal

import cv2, numpy as np
from PIL import Image
import httpx

from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query, Response

app = FastAPI(title="Scan Endpoint")

def deskew_and_binarize(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # grobe Schiefstandskorrektur
    coords = np.column_stack(np.where(gray > 0))
    angle = 0.0
    if coords.size:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # “Scan”-Look
    bw = cv2.adaptiveThreshold(rot, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 15)
    # Ränder weg
    mask = bw < 250
    if mask.any():
        yx = np.argwhere(mask)
        y0, x0 = yx.min(axis=0)
        y1, x1 = yx.max(axis=0) + 1
        bw = bw[y0:y1, x0:x1]
    return bw

async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content

@app.post("/scan")
async def scan(
    request: Request,
    response: Literal["pdf", "json"] = Query("json"),
    file: UploadFile | None = File(None),
):
    # Body nur einmal lesen
    raw = await request.body()
    ctype = (request.headers.get("content-type") or "").lower()

    file_url: Optional[str] = None
    file_b64: Optional[str] = None

    # 1) JSON?
    if "application/json" in ctype or (raw.startswith(b"{") and b"}" in raw[:4096]):
        try:
            obj = json.loads(raw.decode("utf-8"))
            file_url = obj.get("file_url") or obj.get("url") or obj.get("file-url")
            file_b64 = obj.get("file_base64") or obj.get("fileBase64")
        except Exception:
            pass

    # 2) Form / multipart?
    if not (file_url or file_b64 or file) and (
        "multipart/form-data" in ctype or "application/x-www-form-urlencoded" in ctype
    ):
        try:
            form = await request.form()
            file_url = form.get("file_url") or form.get("url") or form.get("file-url")
            file_b64 = form.get("file_base64") or form.get("fileBase64")
        except Exception:
            pass

    # 3) Nur eine URL als Text?
    if not (file_url or file_b64 or file):
        text = raw.decode("utf-8", "ignore").strip()
        if text.startswith("http"):
            file_url = text

    if not (file or file_url or file_b64):
        raise HTTPException(status_code=422, detail="Provide a file or file_url")

    # Bytes beschaffen
    if file is not None:
        data = await file.read()
    elif file_b64:
        data = base64.b64decode(file_b64.split(",")[-1])
    else:
        data = await _download(file_url)

    # Bild dekodieren
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    scanned = deskew_and_binarize(img)

    if response == "pdf":
        # als PDF zurück
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.pagesizes import letter

        rgb = cv2.cvtColor(scanned, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(rgb)
        png = io.BytesIO()
        pil.save(png, format="PNG")
        png.seek(0)

        page_w, page_h = letter
        margin = 36
        avail_w, avail_h = page_w - 2 * margin, page_h - 2 * margin
        h_px, w_px = scanned.shape
        scale = min(avail_w / w_px, avail_h / h_px)
        draw_w, draw_h = w_px * scale, h_px * scale
        x, y = (page_w - draw_w) / 2, (page_h - draw_h) / 2

        out = io.BytesIO()
        c = canvas.Canvas(out, pagesize=letter)
        c.drawImage(ImageReader(png), x, y, width=draw_w, height=draw_h, mask="auto")
        c.showPage()
        c.save()
        pdf = out.getvalue()
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline; filename=scan.pdf"},
        )

    # sonst Base64 (PNG) als JSON
    ok, enc = cv2.imencode(".png", scanned)
    if not ok:
        raise HTTPException(status_code=500, detail="Encode failed")
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    return {"filename": "scan.png", "content_type": "image/png", "file_base64": b64}
