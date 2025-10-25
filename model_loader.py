import torch
from transformers import AutoModelForImageSegmentation, AutoImageProcessor

MODEL_ID = "briaai/RMBG-2.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageSegmentation.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
model.eval()
