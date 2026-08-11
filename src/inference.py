from pathlib import Path
import torch
import json
from model import Autoencoder
from preprocessing import (npz_to_tensor, vertical_segment_scores )
import time 


MODEL_PATH = Path("model/cnn_ae_model.pth")
DATA_DIR = Path("data/chatter")
THRESHOLD_PATH = Path("model/ae_thresholds.json")

device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Select input file
# ---------------------------------------------------------------------------
sample_file = next(DATA_DIR.glob("*.npz"))
print(f"Input file: {sample_file}")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
image, label = npz_to_tensor(sample_file)

print(f"Label: {label}")
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Value range: {image.min():.4f} to {image.max():.4f}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = Autoencoder().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully")

# ---------------------------------------------------------------------------
# Prepare model input
# ---------------------------------------------------------------------------

input_tensor = torch.from_numpy(image)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(
    device=device,
    dtype=torch.float32
)

print(f"Model input shape: {input_tensor.shape}")

# ---------------------------------------------------------------------------
# Run reconstruction
# ---------------------------------------------------------------------------
t1 = time.perf_counter()

with torch.inference_mode():
    reconstruction = model(input_tensor)

print(f"Reconstruction shape: {reconstruction.shape}")
print(
    "Reconstruction range: "
    f"{reconstruction.min().item():.4f} to "
    f"{reconstruction.max().item():.4f}"
)

reconstruction = reconstruction.cpu().numpy()

input_image = input_tensor.cpu().numpy()

t2 = time.perf_counter()

squared_error = (
    input_image - reconstruction
) ** 2

sve_max, sve_topk = vertical_segment_scores(
    squared_error,
    n_segments=10,
    top_k=3
)

t3 = time.perf_counter()

with open(THRESHOLD_PATH, "r") as f:
    thresholds = json.load(f)

threshold = thresholds["sve_topk"]["threshold"]

score = float(sve_topk[0])

prediction = (
    "chatter"
    if score >= threshold
    else "no_chatter"
)

print(f"SVE TopK   : {score:.8f}")
print(f"Threshold  : {threshold:.8f}")
print(f"Prediction : {prediction}")
print(f"Label      : {label}")
print("\nTiming")
print("-" * 30)
print(f"Preprocessing : {(t1 - t0) * 1000:.2f} ms")
print(f"Inference     : {(t2 - t1) * 1000:.2f} ms")
print(f"Scoring       : {(t3 - t2) * 1000:.2f} ms")
print(f"Total         : {(t3 - t0) * 1000:.2f} ms")