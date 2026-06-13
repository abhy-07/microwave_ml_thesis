"""
cross_val.py
============
Volunteer-level GroupKFold cross-validation.

With only 16 volunteers in the September 2022 dataset, any single
train/val split is highly sensitive to which volunteers land in each set.
GroupKFold ensures every volunteer appears in exactly one val fold,
preventing label leakage while giving stable, averaged metrics.

Usage:
    from evaluation.cross_val import cross_val_baselines
    results_df = cross_val_baselines(X, y, meta)
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_COLS, RANDOM_SEED
from data.features import add_band_features


def cross_val_baselines(
    X:          np.ndarray,
    y:          np.ndarray,
    meta:       pd.DataFrame,
    n_splits:   int = 4,
) -> pd.DataFrame:
    """
    GroupKFold cross-validation for RF, XGBoost, FCNN baselines.

    Groups = volunteer_id, so no volunteer appears in both train and val
    within the same fold.

    Args:
        X        : Raw feature matrix (n_samples, 800)
        y        : Label matrix (n_samples, n_targets)
        meta     : DataFrame with 'volunteer_id' column
        n_splits : Number of folds (default 4 → ~4 volunteers per val fold)

    Returns:
        DataFrame with columns: model, target, fold, RMSE, MAE, R2
    """
    from models.baselines import get_rf, get_xgb, get_fcnn
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    target_names = TARGET_COLS[:y.shape[1]]
    groups       = meta["volunteer_id"].values

    gkf = GroupKFold(n_splits=n_splits)

    model_factories = {
        "Random Forest": get_rf,
        "XGBoost":       get_xgb,
        "FCNN":          get_fcnn,
    }

    records = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        train_vols = sorted(set(groups[train_idx]))
        val_vols   = sorted(set(groups[val_idx]))
        print(f"\n  Fold {fold}/{n_splits} | Train vols: {[int(v) for v in train_vols]} | "
              f"Val vols: {[int(v) for v in val_vols]}")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Band features for RF/XGBoost; scaled raw features for FCNN
        X_tr_band  = add_band_features(X_tr)
        X_val_band = add_band_features(X_val)

        scaler     = StandardScaler()
        X_tr_s     = scaler.fit_transform(X_tr)
        X_val_s    = scaler.transform(X_val)

        feature_sets = {
            "Random Forest": (X_tr_band, X_val_band),
            "XGBoost":       (X_tr_band, X_val_band),
            "FCNN":          (X_tr_s,    X_val_s),
        }

        for model_name, factory in model_factories.items():
            model = factory()
            Xtr_m, Xval_m = feature_sets[model_name]

            model.fit(Xtr_m, y_tr)
            y_pred = model.predict(Xval_m)

            for i, tname in enumerate(target_names):
                rmse = float(np.sqrt(mean_squared_error(y_val[:, i], y_pred[:, i])))
                mae  = float(mean_absolute_error(y_val[:, i], y_pred[:, i]))
                r2   = float(r2_score(y_val[:, i], y_pred[:, i]))
                records.append({
                    "model": model_name, "target": tname,
                    "fold": fold, "RMSE": rmse, "MAE": mae, "R2": r2,
                })
                print(f"    {model_name:15s} | {tname:12s} | "
                      f"RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}")

    df = pd.DataFrame(records)
    return df


def summarise_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean ± std across folds for each (model, target) pair.
    """
    summary = (
        cv_df
        .groupby(["model", "target"])[["RMSE", "MAE", "R2"]]
        .agg(["mean", "std"])
        .round(4)
    )
    return summary


def save_cv_summary(cv_df: pd.DataFrame) -> Path:
    from config import METRICS_DIR
    from datetime import datetime

    summary = summarise_cv(cv_df)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    path    = METRICS_DIR / f"baseline_cv_summary_{ts}.txt"

    lines = [
        "Baseline Cross-Validation Summary (GroupKFold on volunteer_id)",
        f"Folds: {cv_df['fold'].nunique()} | Volunteers: {cv_df['fold'].nunique()} groups",
        "=" * 65,
        "",
        summary.to_string(),
        "",
        "Full fold-by-fold results:",
        cv_df.to_string(index=False),
    ]
    path.write_text("\n".join(lines))
    print(f"[cross_val] CV summary saved → {path}")
    return path
