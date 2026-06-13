"""
loader.py
=========
Loads the Viktor MAS Volunteer dataset:
  - Parses S2P (Touchstone) files → S-parameter arrays
  - Loads Excel metadata → tissue thickness labels
  - Merges features + labels into a single DataFrame

Usage:
    from data.loader import load_training_data, load_validation_data

    X_train, y_train, meta_train = load_training_data()
    X_val,   y_val,   meta_val   = load_validation_data()
"""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SEPT_2022_DIR, SEPT_2022_METADATA,
    MARCH_2023_DIR, MARCH_2023_METADATA,
    N_FREQ_POINTS, TARGET_COLS,
)


# ---------------------------------------------------------------------------
# S2P Parsing
# ---------------------------------------------------------------------------

def parse_s2p(filepath: Path) -> Optional[Dict]:
    """
    Parse a Touchstone S2P file.

    Returns a dict with:
        frequency   : np.ndarray [Hz]
        s11_real    : np.ndarray
        s11_imag    : np.ndarray
        s21_real    : np.ndarray
        s21_imag    : np.ndarray
    Returns None if the file cannot be parsed.
    """
    rows = []
    try:
        with open(filepath, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("!") or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        rows.append([float(p) for p in parts[:5]])
                    except ValueError:
                        continue
    except Exception:
        return None

    if len(rows) < 10:
        return None

    arr = np.array(rows)
    return {
        "frequency": arr[:, 0],
        "s11_real":  arr[:, 1],
        "s11_imag":  arr[:, 2],
        "s21_real":  arr[:, 3],
        "s21_imag":  arr[:, 4],
    }


def _volunteer_id_from_path(filepath: Path) -> Optional[int]:
    """Extract volunteer number from a path like '.../Volunteer 3/...'."""
    for part in filepath.parts:
        m = re.match(r"Volunteer\s+(\d+)$", part, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _path_metadata(filepath: Path) -> Dict:
    """Extract device, condition, repetition from file path."""
    parts = filepath.parts
    return {
        "volunteer_id": _volunteer_id_from_path(filepath),
        "device":       parts[-3] if len(parts) >= 3 else "unknown",
        "condition":    parts[-2] if len(parts) >= 2 else "unknown",
        "repetition":   filepath.stem,        # M1, M2, M3
        "filepath":     str(filepath),
    }


# ---------------------------------------------------------------------------
# Feature extraction from one S2P file
# ---------------------------------------------------------------------------

def extract_features(s2p: Dict, n_points: int = N_FREQ_POINTS) -> np.ndarray:
    """
    Convert raw S-parameter arrays into a fixed-length feature vector.

    Strategy:
      1. Subsample the frequency axis to `n_points` evenly-spaced indices.
      2. Extract S11 magnitude (dB), S21 magnitude (dB), S11 phase, S21 phase
         at those points.
      3. Concatenate → feature vector of length 4 × n_points.

    This gives neural networks enough spectral resolution while keeping
    dimensionality manageable (default: 800 features).
    """
    freq = s2p["frequency"]
    total = len(freq)
    indices = np.linspace(0, total - 1, n_points, dtype=int)

    s11_r = s2p["s11_real"][indices]
    s11_i = s2p["s11_imag"][indices]
    s21_r = s2p["s21_real"][indices]
    s21_i = s2p["s21_imag"][indices]

    eps = 1e-10
    s11_mag   = 10 * np.log10(s11_r**2 + s11_i**2 + eps)
    s21_mag   = 10 * np.log10(s21_r**2 + s21_i**2 + eps)
    s11_phase = np.arctan2(s11_i, s11_r)
    s21_phase = np.arctan2(s21_i, s21_r)

    return np.concatenate([s11_mag, s21_mag, s11_phase, s21_phase])


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_sept_2022_labels() -> pd.DataFrame:
    """
    Load and pivot the September 2022 metadata Excel.

    The raw Excel has one row per (volunteer, tissue_type) — 3 rows per volunteer.
    We pivot to wide format: one row per volunteer with columns:
        volunteer_id | Skin_mm | Fat_mm | Muscle_cm2 | Age | BMI | ...

    Prints column names and first rows so you can verify the structure.
    """
    print(f"[loader] Reading: {SEPT_2022_METADATA.name}")
    df = pd.read_excel(SEPT_2022_METADATA)

    print(f"[loader] Raw shape: {df.shape}")
    print(f"[loader] Columns: {df.columns.tolist()}")
    print(df.head(6).to_string())
    print()

    # --- Normalise column names (strip whitespace) ---
    df.columns = df.columns.str.strip()

    # Try to auto-detect key columns
    # Common naming in the Excel based on known structure
    col_map = _detect_label_columns(df)
    print(f"[loader] Detected columns: {col_map}")

    volunteer_col = col_map["volunteer"]
    measure_col   = col_map["measure"]
    value_col     = col_map["value"]

    # The Excel has a grouped multi-row structure: every 3 consecutive rows belong
    # to one volunteer (Skin, Fat, Rfcsa).  The volunteer number only appears in
    # one of those 3 rows.  We fill it across all rows in the same group.
    # Strategy: bfill(limit=1) then ffill(limit=2) propagates the ID to neighbours.
    df[volunteer_col] = (
        df[volunteer_col]
        .bfill(limit=1)
        .ffill(limit=2)
    )

    # Keep only relevant rows (skip rows where measure is NaN)
    df = df.dropna(subset=[measure_col, value_col])

    # Normalise measure names → Skin_mm, Fat_mm, Muscle_cm2
    # Raw values are e.g. "Skin[mm]", "Fat[mm]", "Rfcsa[cm2]"
    df[measure_col] = df[measure_col].str.strip().str.lower()
    measure_rename = {
        "skin":   "Skin_mm",
        "fat":    "Fat_mm",
        "muscle": "Muscle_cm2",
        "rfcsa":  "Muscle_cm2",   # "Rfcsa[cm2]" → Rectus Femoris CSA = muscle proxy
    }
    df[measure_col] = df[measure_col].map(
        lambda x: next((v for k, v in measure_rename.items() if k in x), x)
    )

    # Pivot to wide format
    df_wide = df.pivot_table(
        index=volunteer_col,
        columns=measure_col,
        values=value_col,
        aggfunc="mean",
    ).reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.rename(columns={volunteer_col: "volunteer_id"})

    # Attach demographic columns if present
    demo_cols = [c for c in df.columns
                 if c.lower() in {"age", "gender", "bmi", "height", "weight",
                                   "height[m]", "weight[kg]", "bmi[kg/m2]"}]
    if demo_cols:
        demo = df.drop_duplicates(subset=[volunteer_col])[[volunteer_col] + demo_cols]
        demo = demo.rename(columns={volunteer_col: "volunteer_id"})
        df_wide = df_wide.merge(demo, on="volunteer_id", how="left")

    print(f"[loader] Labels shape after pivot: {df_wide.shape}")
    print(df_wide.head().to_string())
    print()
    return df_wide


def _detect_label_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Heuristically detect volunteer, measure, and value column names."""
    cols_lower = {c.lower(): c for c in df.columns}

    # Volunteer column
    volunteer_col = next(
        (cols_lower[k] for k in cols_lower if "volunteer" in k or "subject" in k),
        df.columns[0],
    )

    # Measure column (Skin / Fat / Muscle)
    measure_col = next(
        (cols_lower[k] for k in cols_lower if "measure" in k or "tissue" in k or "type" in k),
        None,
    )
    if measure_col is None:
        # Fallback: look for a column that contains "skin"/"fat"/"muscle" values
        for c in df.columns:
            sample = df[c].dropna().astype(str).str.lower()
            if sample.str.contains("skin|fat|muscle").any():
                measure_col = c
                break

    # Value column (the numeric measurement — Average or Measure)
    value_col = next(
        (cols_lower[k] for k in cols_lower
         if k in {"average", "value", "thickness", "measurement", "avg"}),
        None,
    )
    if value_col is None:
        # Fallback: last numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_col = numeric_cols[-1] if numeric_cols else df.columns[-1]

    return {
        "volunteer": volunteer_col,
        "measure":   measure_col,
        "value":     value_col,
    }


def load_march_2023_labels() -> pd.DataFrame:
    """
    Load and parse the March 2023 ultrasound-verified ground truth labels.

    The Excel has the same 3-row-per-volunteer structure as September 2022,
    but with two differences:
      1. The volunteer number column is 'Unnamed: 1' (not 'Volunteer number').
      2. There is a known data-entry error: volunteer 15's Fat row has '17.0'
         in the number column instead of '15.0'. To avoid this propagating,
         we assign volunteer IDs sequentially by row-group position (every
         group of 3 consecutive rows = one volunteer, in order).

    Returns a wide-format DataFrame: one row per volunteer with columns
        volunteer_id | Skin_mm | Fat_mm | Muscle_cm2 | Age | ...
    """
    print(f"[loader] Reading: {MARCH_2023_METADATA.name}")
    raw = pd.read_excel(MARCH_2023_METADATA)
    raw.columns = raw.columns.str.strip()

    # Drop the first row (sub-header: Rep_1, Rep_2, Rep_3)
    df = raw.iloc[1:].reset_index(drop=True)

    # Assign volunteer IDs sequentially: rows 0-2 → vol 1, rows 3-5 → vol 2…
    # This is robust to any data-entry errors in the number column.
    df["volunteer_id"] = (df.index // 3) + 1

    # Measure and Average columns
    measure_col = "Measure"
    value_col   = "Average"

    df = df.dropna(subset=[measure_col, value_col])

    # Normalise measure names
    df[measure_col] = df[measure_col].str.strip().str.lower()
    measure_rename = {
        "skin":   "Skin_mm",
        "fat":    "Fat_mm",
        "rfcsa":  "Muscle_cm2",
        "muscle": "Muscle_cm2",
    }
    df[measure_col] = df[measure_col].map(
        lambda x: next((v for k, v in measure_rename.items() if k in x), x)
    )

    # Convert Average to numeric (some cells may have text like "6,1" instead of "6.1")
    df[value_col] = (
        df[value_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    # Pivot to wide format
    df_wide = df.pivot_table(
        index="volunteer_id",
        columns=measure_col,
        values=value_col,
        aggfunc="mean",
    ).reset_index()
    df_wide.columns.name = None

    print(f"[loader] March 2023 labels: {df_wide.shape[0]} volunteers")
    print(df_wide[["volunteer_id", "Skin_mm", "Fat_mm", "Muscle_cm2"]].to_string(index=False))
    print()
    return df_wide


# ---------------------------------------------------------------------------
# Full dataset assembly
# ---------------------------------------------------------------------------

def _load_s2p_dataset(base_dir: Path) -> pd.DataFrame:
    """
    Find all S2P files under `base_dir`, parse each, extract features,
    and return a DataFrame with feature columns + metadata columns.
    """
    files = sorted(base_dir.rglob("*.s2p"))
    print(f"[loader] Found {len(files)} S2P files in: {base_dir.name}")

    records = []
    failed  = 0
    for i, fp in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"[loader]   Processed {i + 1}/{len(files)} files...")

        s2p = parse_s2p(fp)
        if s2p is None:
            failed += 1
            continue

        feat = extract_features(s2p)
        meta = _path_metadata(fp)
        records.append({**meta, **{f"f{j}": v for j, v in enumerate(feat)}})

    print(f"[loader] Parsed {len(records)} files successfully ({failed} failed).")
    return pd.DataFrame(records)


def load_training_data(
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load September 2022 training data.

    Returns:
        X    : Feature matrix (n_samples, n_features)
        y    : Label matrix   (n_samples, 3) — [Skin_mm, Fat_mm, Muscle_cm2]
        meta : DataFrame with volunteer_id, device, condition, repetition, filepath
    """
    labels = load_sept_2022_labels()

    df = _load_s2p_dataset(SEPT_2022_DIR)

    # Drop S2P files that couldn't be linked to a volunteer (e.g. Air References)
    n_raw = len(df)
    df = df.dropna(subset=["volunteer_id"])
    if len(df) < n_raw:
        print(f"[loader] Dropped {n_raw - len(df)} files with no volunteer ID (reference folders).")

    if limit:
        df = df.head(limit)

    # Merge labels on volunteer_id
    merged = df.merge(labels, on="volunteer_id", how="inner")
    n_before = len(df)
    n_after  = len(merged)
    if n_after < n_before:
        print(f"[loader] Warning: {n_before - n_after} S2P files had no matching label.")

    feature_cols = [c for c in merged.columns if c.startswith("f") and c[1:].isdigit()]
    label_cols   = [c for c in TARGET_COLS if c in merged.columns]

    if not label_cols:
        raise ValueError(
            f"No target columns found after merge.\n"
            f"Available columns: {merged.columns.tolist()}\n"
            "Check load_sept_2022_labels() output above and update TARGET_COLS in config.py."
        )

    # Drop samples where any target label is missing
    complete = merged.dropna(subset=label_cols)
    n_dropped = len(merged) - len(complete)
    if n_dropped:
        print(f"[loader] Dropped {n_dropped} samples with incomplete target labels.")
    merged = complete

    X    = merged[feature_cols].values.astype(np.float32)
    y    = merged[label_cols].values.astype(np.float32)
    meta = merged[["volunteer_id", "device", "condition", "repetition", "filepath"]].reset_index(drop=True)

    print(f"[loader] Training data ready: X={X.shape}, y={y.shape}")
    print(f"[loader] Targets: {label_cols}")
    return X, y, meta


def load_validation_data() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load March 2023 independent validation data (ultrasound-verified labels).

    Key differences from September 2022:
      - 23 volunteers (not 18)
      - Only 2 devices: nanoVNA, CopperMountain (no miniVNA)
      - Conditions: Bandstop S1, Bandstop S2, Beamer (SRR replaced by Beamer)
      - Air.s2p files inside volunteer folders — excluded (calibration, not tissue)
      - Data-entry error in Excel handled by sequential volunteer ID assignment

    Returns:
        X    : Feature matrix (n_samples, 800) — same feature space as training
        y    : Label matrix   (n_samples, 3)   — [Skin_mm, Fat_mm, Muscle_cm2]
        meta : DataFrame with volunteer_id, device, condition, repetition, filepath
    """
    labels = load_march_2023_labels()
    df     = _load_s2p_dataset(MARCH_2023_DIR)

    # Drop Air.s2p calibration files — their repetition stem is "Air"
    n_before = len(df)
    df = df[df["repetition"] != "Air"]
    print(f"[loader] Dropped {n_before - len(df)} Air.s2p calibration files.")

    # Drop files with no volunteer ID
    n_before = len(df)
    df = df.dropna(subset=["volunteer_id"])
    if len(df) < n_before:
        print(f"[loader] Dropped {n_before - len(df)} files with no volunteer ID.")

    # Merge with labels
    merged = df.merge(labels, on="volunteer_id", how="inner")
    n_dropped = len(df) - len(merged)
    if n_dropped:
        print(f"[loader] Warning: {n_dropped} S2P files had no matching label.")

    feature_cols = [c for c in merged.columns if c.startswith("f") and c[1:].isdigit()]
    label_cols   = [c for c in TARGET_COLS if c in merged.columns]

    if not label_cols:
        raise ValueError(
            f"No target columns found after merge. Available: {merged.columns.tolist()}"
        )

    # Drop rows with any missing target
    complete = merged.dropna(subset=label_cols)
    if len(complete) < len(merged):
        print(f"[loader] Dropped {len(merged) - len(complete)} rows with incomplete labels.")
    merged = complete

    X    = merged[feature_cols].values.astype(np.float32)
    y    = merged[label_cols].values.astype(np.float32)
    meta = merged[["volunteer_id", "device", "condition", "repetition", "filepath"]].reset_index(drop=True)

    print(f"[loader] Validation data ready: X={X.shape}, y={y.shape}")
    print(f"[loader] Targets: {label_cols}")
    print(f"[loader] Unique volunteers: {meta['volunteer_id'].nunique()}")
    return X, y, meta


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SANITY CHECK — loader.py")
    print("=" * 60)

    # Test parsing one S2P file
    test_files = list(SEPT_2022_DIR.rglob("*.s2p"))
    if test_files:
        sample = test_files[0]
        print(f"\nParsing sample file: {sample.name}")
        s2p = parse_s2p(sample)
        if s2p:
            print(f"  Frequency points : {len(s2p['frequency'])}")
            print(f"  Freq range       : {s2p['frequency'][0]/1e9:.3f} – {s2p['frequency'][-1]/1e9:.3f} GHz")
            feat = extract_features(s2p)
            print(f"  Feature vector   : length {len(feat)}")
        else:
            print("  ERROR: could not parse file.")

    # Load labels
    print("\n--- September 2022 Labels ---")
    labels = load_sept_2022_labels()
    print(labels)
