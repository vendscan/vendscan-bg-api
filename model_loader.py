from transformers import AutoModelForImageSegmentation, AutoImageProcessor
import torch

# Use the ungated, lightweight model
MODEL_ID = "briaai/RMBG-1.4"

# Pick CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor + model
processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageSegmentation.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
model.eval()
