# app.py — FastAPI + RMBG-1.4 transparent background remover
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from model_loader import model, device

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Load image into Pillow
    try:
        image = Image.open(file.file).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Could not load image")

    W, H = image.size

    # --- preprocess (same as your working local code) ---
    img_np = np.array(image).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        out = model(img_t)

    # extract logits
    logits = out[0] if isinstance(out, (list, tuple)) else out.logits
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)

    # sigmoid + resize
    prob = torch.sigmoid(logits)
    prob_up = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    alpha = prob_up[0, 0].clamp(0, 1).cpu().numpy()

    # create transparent PNG
    alpha_8 = (alpha * 255).astype(np.uint8)
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha_8, mode="L"))

    # save to bytes for response
    from io import BytesIO
    buffer = BytesIO()
    rgba.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(buffer.getvalue(), media_type="image/png")
