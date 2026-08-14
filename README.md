# anomaly-ae-deployment

Deployment repository for running and benchmarking a CNN-based autoencoder for spectrogram anomaly detection.

The pipeline converts windowed time series data from `.npz` files into RGB spectrograms, runs the trained autoencoder, computes the reconstruction error and classifies each sample as `chatter` or `no_chatter`.

## Pipeline

```text
NPZ time series
→ Spectrograms X/Y/Z
→ RGB spectrogram 150x100
→ CNN autoencoder
→ Reconstruction error
→ SVE TopK score
→ Threshold decision
```

## Build the image:
```cmd
docker build --no-cache -t anomaly-ae .
```
## Run Single Inference
```python
python src/inference.py
```
Runs the complete pipeline for a single .npz file and prints:

- label
- prediction
- anomaly score
- threshold
- timing information

## Run the benchmark / Start from Repo-Root directory:
```cmd
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  anomaly-ae
```
Processes all .npz files located in the data/ directory

## Purpose

The repository is intended to benchmark the complete inference workflow on edge devices such as the Revolution Pi and compare preprocessing and inference runtimes across different hardware platforms.