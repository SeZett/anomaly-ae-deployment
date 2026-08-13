from pathlib import Path
import json
import time
import platform
import numpy as np
import pandas as pd
import torch

from model import Autoencoder
from preprocessing import (
    npz_to_tensor,
    vertical_segment_scores
)


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = Path("model/cnn_ae_model.pth")
THRESHOLD_PATH = Path("model/ae_thresholds.json")

DATA_ROOT = Path("data")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device("cpu")


# ============================================================================
# Load threshold
# ============================================================================

with open(THRESHOLD_PATH, "r") as f:
    thresholds = json.load(f)

threshold = thresholds["sve_topk"]["threshold"]


# ============================================================================
# Load model
# ============================================================================

model = Autoencoder().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully")


# ============================================================================
# Collect NPZ files
# ============================================================================

npz_files = sorted(DATA_ROOT.rglob("*.npz"))

if not npz_files:
    raise FileNotFoundError(
        f"No NPZ files found below {DATA_ROOT.resolve()}"
    )

print(f"Found {len(npz_files)} NPZ files")

# ============================================================================
# Warmup
# ============================================================================

warmup_file = npz_files[0]

image, _ = npz_to_tensor(warmup_file)

input_tensor = torch.from_numpy(image)
input_tensor = input_tensor.unsqueeze(0).to(
    device=device,
    dtype=torch.float32
)

with torch.inference_mode():
    _ = model(input_tensor)

print("Warmup completed")


# ============================================================================
# Benchmark loop
# ============================================================================

results = []

for idx, npz_file in enumerate(npz_files, start=1):

    print(
        f"[{idx}/{len(npz_files)}] "
        f"Processing {npz_file.name}"
    )

    # ------------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------------

    t0 = time.perf_counter()

    image, label = npz_to_tensor(npz_file)

    input_tensor = torch.from_numpy(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(
        device=device,
        dtype=torch.float32
    )

    t1 = time.perf_counter()

    # ------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------

    with torch.inference_mode():
        reconstruction = model(input_tensor)

    reconstruction = reconstruction.cpu().numpy()
    input_image = input_tensor.cpu().numpy()

    t2 = time.perf_counter()

    # ------------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------------

    squared_error = (
        input_image - reconstruction
    ) ** 2

    sve_max, sve_topk = vertical_segment_scores(
        squared_error,
        n_segments=10,
        top_k=3
    )

    score = float(sve_topk[0])

    prediction = (
        "chatter"
        if score >= threshold
        else "no_chatter"
    )

    t3 = time.perf_counter()

    # ------------------------------------------------------------------------
    # Store result
    # ------------------------------------------------------------------------

    results.append({
        "file": npz_file.name,
        "label": label,
        "prediction": prediction,
        "correct": prediction == label,
        "score": score,
        "preprocessing_ms": (t1 - t0) * 1000,
        "inference_ms": (t2 - t1) * 1000,
        "scoring_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
    })


# ============================================================================
# Results
# ============================================================================

results_df = pd.DataFrame(results)

output_csv = RESULTS_DIR / "benchmark_results.csv"
results_df.to_csv(output_csv, index=False)

accuracy = (
    results_df["correct"].mean()
    if len(results_df) > 0
    else 0.0
)

print("\n")
print("=" * 60)
print("Benchmark Summary")
print("=" * 60)

print(f"Samples             : {len(results_df)}")
print(f"Accuracy            : {accuracy:.4f}")

print(
    f"Mean preprocessing  : "
    f"{results_df['preprocessing_ms'].mean():.2f} ms"
)

print(
    f"Mean inference      : "
    f"{results_df['inference_ms'].mean():.2f} ms"
)

print(
    f"Mean scoring        : "
    f"{results_df['scoring_ms'].mean():.2f} ms"
)

print(
    f"Mean total          : "
    f"{results_df['total_ms'].mean():.2f} ms"
)

print(
    f"Median total        : "
    f"{results_df['total_ms'].median():.2f} ms"
)

print(
    f"Max total           : "
    f"{results_df['total_ms'].max():.2f} ms"
)

print()
print(f"Results written to: {output_csv}")
print(platform.platform())
print(torch.__version__)