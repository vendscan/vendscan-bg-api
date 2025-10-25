import io, zipfile, os
from typing import List
from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

from model_loader import processor, model, device

# Load API key from environment
API_KEY = os.getenv("API_KEY")

app = FastAPI(title="VendScan Background Removal API", version="1.0.0")

# CORS (tighten later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

def verify_key(key: str):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="Server missing API_KEY env")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

def process_image_to_png_bytes(img_bytes: bytes) -> bytes:
    # Load image
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = image.size

    # Preprocess (HF logic)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        out = model(pixel_values)

    # Extract logits (HF behavior)
    if isinstance(out, (list, tuple)):
        logits = out[-1]
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out

    if logits.dim() == 3:
        logits = logits.unsqueeze(1)

    # Matte → mask → transparent RGBA
    matte = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = Image.fromarray((matte * 255).astype("uint8"), "L").resize((W, H), Image.LANCZOS)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)

    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()

@app.post("/remove-bg", response_class=Response)
async def remove_bg(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    verify_key(x_api_key)
    data = await file.read()
    png_bytes = process_image_to_png_bytes(data)
    return Response(content=png_bytes, media_type="image/png")

@app.post("/remove-bg-batch", response_class=Response)
async def remove_bg_batch(
    files: List[UploadFile] = File(...),
    x_api_key: str = Header(None)
):
    verify_key(x_api_key)

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            try:
                data = await f.read()
                png_bytes = process_image_to_png_bytes(data)
                name = f.filename.rsplit(".",1)[0] + ".png"
                zf.writestr(name, png_bytes)
            except Exception as e:
                zf.writestr(f"{f.filename}_ERROR.txt", str(e))

    zip_buf.seek(0)
    return Response(content=zip_buf.getvalue(), media_type="application/zip")
