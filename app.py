import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from model_loader import processor, model, device
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import io

API_KEY = os.getenv("API_KEY")

app = FastAPI()

def remove_bg_on_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = image.size

    inputs = processor(image, return_tensors="pt")
    pixel_values = next(v for v in inputs.values() if isinstance(v, torch.Tensor)).to(device)

    with torch.no_grad():
        out = model(pixel_values)

    logits = out[0] if isinstance(out, (list, tuple)) else out.logits
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)

    prob = torch.sigmoid(logits)
    prob_up = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    alpha = prob_up[0, 0].clamp(0, 1).cpu().numpy()

    mask_8u = (alpha * 255).astype(np.uint8)
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask_8u, mode="L"))

    output = io.BytesIO()
    rgba.save(output, format="PNG")
    output.seek(0)
    return output

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...), x_api_key: str = None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    image_bytes = await file.read()
    output = remove_bg_on_image(image_bytes)
    return StreamingResponse(output, media_type="image/png")

@app.post("/remove-bg-batch")
async def remove_bg_batch(files: list[UploadFile] = File(...), x_api_key: str = None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    results = []
    for f in files:
        image_bytes = await f.read()
        output = remove_bg_on_image(image_bytes)
        results.append(output.getvalue())

    return {"results": [res.hex() for res in results]}  # client will decode hex to PNG bytes
