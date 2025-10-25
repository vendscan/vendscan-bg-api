# model_loader.py — for RMBG-1.4
import torch
from transformers import AutoModelForImageSegmentation

MODEL_ID = "briaai/RMBG-1.4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model only (RMBG-1.4 does not use AutoImageProcessor)
model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
).to(device)

model.eval()
