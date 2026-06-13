"""
features.py
===========
Band-statistical feature engineering for RF/XGBoost models.

While loader.py produces a 4×N_FREQ_POINTS raw feature vector (good for neural nets),
tree-based models often work better with hand-crafted, interpretable features.

This module adds a second feature representation:
  For each of the 4 signal channels (S11_mag, S21_mag, S11_phase, S21_phase)
  and each frequency sub-band, compute: mean, std, min, max, slope.
  → 4 channels × N_FREQ_BANDS bands × 5 stats = 200 features (default).

Usage:
    from data.features import add_band_features, get_band_feature_names

    X_band = add_band_features(X_raw)    # X_raw from loader.load_training_data()
    names  = get_band_feature_names()
"""

import numpy as np
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import N_FREQ_POINTS, N_FREQ_BANDS


# Feature vector layout from loader.py:
#   [0 : N]   S11 magnitude (dB)
#   [N : 2N]  S21 magnitude (dB)
#   [2N: 3N]  S11 phase (rad)
#   [3N: 4N]  S21 phase (rad)
_N = N_FREQ_POINTS
_CHANNELS = ["S11_mag", "S21_mag", "S11_phase", "S21_phase"]
_CHANNEL_SLICES = [
    slice(0,    _N),
    slice(_N,   2 * _N),
    slice(2*_N, 3 * _N),
    slice(3*_N, 4 * _N),
]
_STATS = ["mean", "std", "min", "max", "slope"]


def _band_stats(signal: np.ndarray, n_bands: int = N_FREQ_BANDS) -> np.ndarray:
    """Compute [mean, std, min, max, slope] for each of `n_bands` sub-bands."""
    band_size = len(signal) // n_bands
    features = []
    for b in range(n_bands):
        seg = signal[b * band_size : (b + 1) * band_size]
        if len(seg) == 0:
            features.extend([0.0] * len(_STATS))
            continue
        x = np.arange(len(seg), dtype=float)
        slope = np.polyfit(x, seg, 1)[0] if len(seg) > 1 else 0.0
        features.extend([
            float(np.mean(seg)),
            float(np.std(seg)),
            float(np.min(seg)),
            float(np.max(seg)),
            float(slope),
        ])
    return np.array(features, dtype=np.float32)


def add_band_features(X: np.ndarray, n_bands: int = N_FREQ_BANDS) -> np.ndarray:
    """
    Transform raw feature matrix X (n_samples, 4*N_FREQ_POINTS)
    into band-statistical features (n_samples, 4 * n_bands * 5).

    Args:
        X      : Raw feature matrix from loader.load_training_data()
        n_bands: Number of frequency sub-bands

    Returns:
        X_band : Band-statistical feature matrix
    """
    result = []
    for row in X:
        row_feats = []
        for sl in _CHANNEL_SLICES:
            row_feats.append(_band_stats(row[sl], n_bands))
        result.append(np.concatenate(row_feats))
    return np.array(result, dtype=np.float32)


def get_band_feature_names(n_bands: int = N_FREQ_BANDS) -> List[str]:
    """Return human-readable names for each band-statistical feature."""
    names = []
    for ch in _CHANNELS:
        for b in range(n_bands):
            for stat in _STATS:
                names.append(f"{ch}_band{b}_{stat}")
    return names


def add_global_stats(X: np.ndarray) -> np.ndarray:
    """
    Add global (whole-spectrum) statistics as extra features.
    Appended to the right of X.

    Adds: mean, std, min, max, peak frequency index for each channel.
    → 5 × 4 = 20 extra features.
    """
    extras = []
    for row in X:
        row_extra = []
        for sl in _CHANNEL_SLICES:
            seg = row[sl]
            row_extra.extend([
                np.mean(seg),
                np.std(seg),
                np.min(seg),
                np.max(seg),
                float(np.argmax(np.abs(seg))) / len(seg),  # normalised peak position
            ])
        extras.append(row_extra)
    return np.hstack([X, np.array(extras, dtype=np.float32)])


if __name__ == "__main__":
    # Quick self-test with random data
    X_dummy = np.random.randn(10, 4 * N_FREQ_POINTS).astype(np.float32)
    X_band  = add_band_features(X_dummy)
    names   = get_band_feature_names()
    print(f"Raw features  : {X_dummy.shape}")
    print(f"Band features : {X_band.shape}")
    print(f"Feature names : {len(names)} names")
    print(f"First 5 names : {names[:5]}")
