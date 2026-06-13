"""
config.py
=========
Central configuration for all paths and hyperparameters.
All other modules import from here — never hardcode paths elsewhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root directories
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Datafiles" / "Viktor"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
SEPT_2022_DIR      = DATA_DIR / "MAS Volunteer Study September 2022"
MARCH_2023_DIR     = DATA_DIR / "MAS Volunteer Study March 2023"
PHANTOM_DIR        = DATA_DIR / "Femoral Head Measurements 050520 and 060520"

SEPT_2022_METADATA = SEPT_2022_DIR / "anonimized_volunteer_measurements_uppsala_september_complete.xlsx"
MARCH_2023_METADATA = MARCH_2023_DIR / "Volunteer_trial_US_data.xlsx"

# ---------------------------------------------------------------------------
# Output directories (auto-created)
# ---------------------------------------------------------------------------
OUTPUT_DIR  = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR  = OUTPUT_DIR / "models"

for _d in [FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature extraction settings
# ---------------------------------------------------------------------------
# Number of evenly-spaced frequency points to subsample from the full sweep
# (~2020 points → N_FREQ_POINTS used as raw features for neural nets)
N_FREQ_POINTS = 200

# Number of frequency sub-bands for statistical features (used by RF/XGBoost)
N_FREQ_BANDS = 10

# Target tissue columns (must match pivoted Excel column names)
TARGET_COLS = ["Skin_mm", "Fat_mm", "Muscle_cm2"]

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
RANDOM_SEED  = 42
TEST_SIZE    = 0.20   # Fraction of training data held out for dev evaluation
VAL_SPLIT    = 0.15   # Fraction of training data used for validation during training

# MC Dropout
MC_T         = 50     # Number of stochastic forward passes at inference
DROPOUT_P    = 0.3

# Neural net training
BATCH_SIZE   = 32
MAX_EPOCHS   = 300
LR           = 1e-3
PATIENCE     = 20     # Early stopping patience

# Deep Ensembles
N_ENSEMBLE   = 5      # Number of independently trained models
