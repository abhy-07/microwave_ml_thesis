"""
baselines.py
============
Deterministic baseline regressors for body composition prediction.

Models:
  - Random Forest Regressor
  - XGBoost Regressor
  - Fully Connected Neural Network Regressor (sklearn MLP)

All models do multi-output regression: predict [Skin_mm, Fat_mm, Muscle_cm2]
from a single feature vector.

Usage:
    from models.baselines import get_rf, get_xgb, get_fcnn, train_eval_baseline

    model = get_rf()
    results = train_eval_baseline("Random Forest", model, X_train, X_test, y_train, y_test)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RANDOM_SEED, METRICS_DIR, TARGET_COLS


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def get_rf(random_state: int = RANDOM_SEED) -> RandomForestRegressor:
    """Random Forest — multi-output regression natively supported."""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )


def get_xgb(random_state: int = RANDOM_SEED) -> MultiOutputRegressor:
    """XGBoost — wrapped in MultiOutputRegressor for multi-target support."""
    base = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    return MultiOutputRegressor(base, n_jobs=-1)


def get_fcnn(random_state: int = RANDOM_SEED) -> MLPRegressor:
    """Fully Connected Neural Network — sklearn MLP for multi-output regression."""
    return MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=random_state,
        learning_rate_init=1e-3,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names=None) -> pd.DataFrame:
    """
    Compute per-target RMSE, MAE, R² and return a tidy DataFrame.

    Args:
        y_true       : (n_samples, n_targets)
        y_pred       : (n_samples, n_targets)
        target_names : list of column names (default: from config.TARGET_COLS)
    """
    if target_names is None:
        target_names = TARGET_COLS[:y_true.shape[1]]

    rows = []
    for i, name in enumerate(target_names):
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        mae  = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        r2   = float(r2_score(y_true[:, i], y_pred[:, i]))
        rows.append({"target": name, "RMSE": rmse, "MAE": mae, "R2": r2})
    return pd.DataFrame(rows)


def train_eval_baseline(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    target_names=None,
) -> Dict:
    """
    Train a baseline model and evaluate it.

    Returns a dict with:
        model       : fitted model
        metrics     : pd.DataFrame with per-target RMSE / MAE / R²
        y_pred      : predictions on X_test
    """
    if target_names is None:
        target_names = TARGET_COLS[:y_train.shape[1]]

    print(f"\n[baselines] Training {model_name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred, target_names)

    print(f"[baselines] {model_name} results:")
    print(metrics.to_string(index=False))

    return {"model": model, "metrics": metrics, "y_pred": y_pred}


def save_baseline_metrics(all_results: Dict[str, Dict], target_names=None) -> Path:
    """
    Save all baseline results to a timestamped text file in outputs/metrics/.

    Args:
        all_results  : {model_name: result_dict from train_eval_baseline()}
        target_names : list of target column names
    """
    if target_names is None:
        target_names = TARGET_COLS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = METRICS_DIR / f"baseline_metrics_{timestamp}.txt"

    lines = [
        "Uncertainty-Aware ML — Microwave Body Composition",
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Task: Regression → {target_names}",
        "=" * 60,
        "",
    ]

    for model_name, result in all_results.items():
        lines.append(f"Model: {model_name}")
        lines.append(result["metrics"].to_string(index=False))
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"[baselines] Metrics saved → {out_path}")
    return out_path
