# src/preprocessing.py

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import spectrogram


# ============================================================================
# Configuration
# ============================================================================

FS = 50_000

NPERSEG = 512
MAX_FREQ_CUT = 5000

DB_MIN = -75
DB_MAX = -20

EPS = 1e-12

IMAGE_SIZE = (150, 100)


# ============================================================================
# Spectrogram generation
# ============================================================================

def create_spectrogram(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Create a normalized spectrogram identical to the training pipeline.

    Parameters
    ----------
    signal : np.ndarray
        Time series signal.
    fs : int
        Sampling frequency.

    Returns
    -------
    np.ndarray
        Spectrogram normalized to [0, 1].
    """

    noverlap = int(0.75 * NPERSEG)

    frequencies, times, sxx = spectrogram(
        signal,
        fs=fs,
        nperseg=NPERSEG,
        noverlap=noverlap,
        mode="psd"
    )

    mask = frequencies <= MAX_FREQ_CUT
    sxx = sxx[mask, :]

    sxx_db = 10 * np.log10(sxx + EPS)
    sxx_db = np.clip(sxx_db, DB_MIN, DB_MAX)

    sxx_norm = 1.0 - (sxx_db - DB_MIN) / (DB_MAX - DB_MIN)

    return sxx_norm


# ============================================================================
# RGB spectrogram generation
# ============================================================================

def create_rgb_spectrogram(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fs: int = FS
) -> np.ndarray:
    """
    Create RGB spectrogram image from X/Y/Z channels.

    Returns
    -------
    np.ndarray
        RGB image in HWC format with values in range [0, 1].
    """

    spec_x = create_spectrogram(x, fs)
    spec_y = create_spectrogram(y, fs)
    spec_z = create_spectrogram(z, fs)

    rgb = np.stack(
        [spec_x, spec_y, spec_z],
        axis=-1
    )

    rgb = np.clip(rgb, 0.0, 1.0)

    # identical to training pipeline
    rgb = np.flipud(rgb)

    return rgb


# ============================================================================
# Image preparation
# ============================================================================

def rgb_to_tensor(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB spectrogram to model input tensor.

    Returns
    -------
    np.ndarray
        Tensor in CHW format:
        (3, 100, 150)
    """

    image = Image.fromarray(
        (rgb * 255).astype(np.uint8)
    )

    image = image.resize(IMAGE_SIZE)

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    return image


# ============================================================================
# NPZ processing
# ============================================================================

def load_npz(npz_path: str | Path):
    """
    Load X, Y, Z signals from npz file.
    """

    npz = np.load(npz_path)

    x = npz["X"]
    y = npz["Y"]
    z = npz["Z"]

    label = None

    if "label" in npz.files:
        label = str(npz["label"])

    return x, y, z, label


def npz_to_tensor(
    npz_path: str | Path,
    fs: int = FS
) -> tuple[np.ndarray, str | None]:
    """
    Complete preprocessing pipeline.

    NPZ
      -> RGB spectrogram
      -> resize
      -> CHW tensor

    Returns
    -------
    tuple
        (
            tensor shape (3,100,150),
            label
        )
    """

    x, y, z, label = load_npz(npz_path)

    rgb = create_rgb_spectrogram(
        x,
        y,
        z,
        fs=fs
    )

    tensor = rgb_to_tensor(rgb)

    return tensor, label

# ============================================================================
# Vertical segment scoring
# ============================================================================

def vertical_segment_scores(error_map: np.ndarray, n_segments: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    width = error_map.shape[3]
    boundaries = np.linspace(0, width, n_segments + 1, dtype=int)
    segment_scores = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        segment_scores.append(error_map[:, :, :, left:right].mean(axis=(1, 2, 3)))
    segment_scores = np.stack(segment_scores, axis=1)
    sve_max = segment_scores.max(axis=1)
    top_k = min(top_k, segment_scores.shape[1])
    sve_topk = np.sort(segment_scores, axis=1)[:, -top_k:].mean(axis=1)
    return sve_max, sve_topk