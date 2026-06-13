"""
run_pipeline.py
===============
Single entry point for the full thesis pipeline.

Usage (run from the Thesis/ root or from src/):
    python src/run_pipeline.py --stage data
    python src/run_pipeline.py --stage baseline
    python src/run_pipeline.py --stage mc_dropout
    python src/run_pipeline.py --stage ensemble
    python src/run_pipeline.py --stage compare
    python src/run_pipeline.py --stage all

Stages:
    data       → Load + cache Viktor training data; print summary
    baseline   → RF, XGBoost, FCNN (sklearn); save metrics + plots
    mc_dropout → PyTorch MC Dropout regressor; save uncertainty metrics + plots
    ensemble   → Deep Ensembles (N members); save uncertainty metrics + plots
    compare    → Side-by-side comparison of all models
    all        → Run data → baseline → mc_dropout → ensemble → compare

Data split strategy:
    The dataset has 18 volunteers. Because ALL files from the same volunteer
    share the same label, a random file-level split would cause data leakage
    (train and test would contain files from the same volunteer).
    We use a VOLUNTEER-LEVEL split: volunteers 1–14 → train (78%),
    volunteers 15–18 → internal val (22%). The March 2023 dataset will serve
    as the independent held-out test set in a later stage.
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from config import (
    RANDOM_SEED, OUTPUT_DIR, METRICS_DIR, TARGET_COLS,
    BATCH_SIZE, MAX_EPOCHS, LR, PATIENCE, MC_T, N_ENSEMBLE,
)
from data.loader import load_training_data, load_validation_data
from data.features import add_band_features, get_band_feature_names


# ---------------------------------------------------------------------------
# Volunteer-level train/val split
# ---------------------------------------------------------------------------

def volunteer_split(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    val_fraction: float = 0.22,
) -> tuple:
    """
    Split at the volunteer level — all files of one volunteer go to either
    train OR val, never both. This prevents label leakage.

    Selects the top `val_fraction` volunteers (by ID) as the val set.
    With 18 volunteers and val_fraction=0.22 → volunteers 15–18 are val.
    """
    vols = sorted(meta["volunteer_id"].dropna().unique())
    n_val = max(1, round(len(vols) * val_fraction))
    val_vols  = set(vols[-n_val:])
    train_vols = set(vols) - val_vols

    train_mask = meta["volunteer_id"].isin(train_vols).values
    val_mask   = meta["volunteer_id"].isin(val_vols).values

    print(f"[split] Train volunteers: {sorted(train_vols)}  ({train_mask.sum()} files)")
    print(f"[split] Val   volunteers: {sorted(val_vols)}   ({val_mask.sum()} files)")

    return (
        X[train_mask], X[val_mask],
        y[train_mask], y[val_mask],
        meta[train_mask].reset_index(drop=True),
        meta[val_mask].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Stage 1 — Data
# ---------------------------------------------------------------------------

def stage_data(limit=None):
    print("\n" + "=" * 60)
    print("STAGE: DATA LOADING")
    print("=" * 60)

    X, y, meta = load_training_data(limit=limit)

    print(f"\nFeature matrix X : {X.shape}")
    print(f"Label matrix y   : {y.shape}  columns={TARGET_COLS[:y.shape[1]]}")
    print(f"Unique volunteers: {meta['volunteer_id'].nunique()}")

    labels_df = pd.DataFrame(y, columns=TARGET_COLS[:y.shape[1]])
    print(f"\nLabel statistics:\n{labels_df.describe().to_string()}")

    cache = OUTPUT_DIR / "processed_data.pkl"
    with open(cache, "wb") as f:
        pickle.dump({"X": X, "y": y, "meta": meta}, f)
    print(f"\n[pipeline] Data cached → {cache}")
    return X, y, meta


def _load_data():
    cache = OUTPUT_DIR / "processed_data.pkl"
    if cache.exists():
        print(f"[pipeline] Loading cached data.")
        with open(cache, "rb") as f:
            d = pickle.load(f)
        return d["X"], d["y"], d["meta"]
    return stage_data()


# ---------------------------------------------------------------------------
# Stage 2 — Baselines
# ---------------------------------------------------------------------------

def stage_baseline():
    print("\n" + "=" * 60)
    print("STAGE: BASELINE MODELS (volunteer-level split)")
    print("=" * 60)

    from models.baselines import get_rf, get_xgb, get_fcnn, train_eval_baseline, save_baseline_metrics
    from evaluation.plots import plot_predictions_vs_truth, plot_feature_importance, save_fig

    X, y, meta = _load_data()
    target_names = TARGET_COLS[:y.shape[1]]

    X_train, X_val, y_train, y_val, _, _ = volunteer_split(X, y, meta)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    X_train_b = add_band_features(X_train)
    X_val_b   = add_band_features(X_val)
    band_names = get_band_feature_names()

    models = {
        "Random Forest": (get_rf(),  X_train_b, X_val_b),
        "XGBoost":       (get_xgb(), X_train_b, X_val_b),
        "FCNN":          (get_fcnn(), X_train_s, X_val_s),
    }

    all_results = {}
    for name, (model, Xtr, Xte) in models.items():
        result = train_eval_baseline(name, model, Xtr, Xte, y_train, y_val, target_names)
        all_results[name] = result

    save_baseline_metrics(all_results, target_names)

    for model_name, result in all_results.items():
        for i, tname in enumerate(target_names):
            fig = plot_predictions_vs_truth(
                y_val[:, i], result["y_pred"][:, i],
                target_name=tname,
                units="mm" if "mm" in tname else "cm²",
            )
            save_fig(fig, f"baseline_{model_name.lower().replace(' ', '_')}_{tname}")

    rf_model = all_results["Random Forest"]["model"]
    if hasattr(rf_model, "feature_importances_"):
        fig = plot_feature_importance(
            rf_model.feature_importances_, feature_names=band_names, top_n=20,
            title="Random Forest — Top 20 Feature Importances",
        )
        save_fig(fig, "rf_feature_importance")

    # Cache scaler for uncertainty models
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return all_results


# ---------------------------------------------------------------------------
# Stage 2b — Cross-Validated Baselines (proper evaluation)
# ---------------------------------------------------------------------------

def stage_cv():
    print("\n" + "=" * 60)
    print("STAGE: BASELINE CROSS-VALIDATION (GroupKFold on volunteer_id)")
    print("=" * 60)

    from evaluation.cross_val import cross_val_baselines, save_cv_summary

    X, y, meta = _load_data()

    print(f"[cv] Running 4-fold GroupKFold on {meta['volunteer_id'].nunique()} volunteers ...")
    cv_df = cross_val_baselines(X, y, meta, n_splits=4)

    print("\n[cv] Cross-validation summary (mean ± std across folds):")
    from evaluation.cross_val import summarise_cv
    summary = summarise_cv(cv_df)
    print(summary.to_string())

    save_cv_summary(cv_df)
    return cv_df


# ---------------------------------------------------------------------------
# Stage 3 — MC Dropout
# ---------------------------------------------------------------------------

def stage_mc_dropout():
    print("\n" + "=" * 60)
    print("STAGE: MC DROPOUT (volunteer-level split)")
    print("=" * 60)

    from models.mc_dropout import (
        MCDropoutRegressor, train, mc_predict,
        save_model, save_learning_curves, save_uncertainty_plots, save_metrics,
    )

    X, y, meta = _load_data()
    target_names = TARGET_COLS[:y.shape[1]]

    X_train, X_val, y_train, y_val, _, _ = volunteer_split(X, y, meta)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)

    input_dim = X_train_s.shape[1]
    n_targets = y_train.shape[1]

    print(f"\n[mc_dropout] Input dim: {input_dim} | Targets: {n_targets} | Device: ", end="")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = MCDropoutRegressor(
        input_dim=input_dim,
        hidden_dims=(256, 128, 64),
        n_targets=n_targets,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[mc_dropout] Trainable parameters: {total_params:,}")

    print("\n[mc_dropout] Training ...")
    history = train(
        model, X_train_s, y_train, X_val_s, y_val,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=PATIENCE,
        verbose=True,
    )

    print(f"\n[mc_dropout] Running inference (T={MC_T} passes) ...")
    results = mc_predict(model, X_val_s, T=MC_T)

    print("\n[mc_dropout] Saving outputs ...")
    save_model(model, run_tag="mc_dropout")
    save_learning_curves(history, run_tag="mc_dropout")
    save_uncertainty_plots(results, y_val, target_names, run_tag="mc_dropout")
    save_metrics(
        results, y_val, target_names, run_tag="mc_dropout",
        extra_info=f"Train vols: 1-14 | Val vols: 15-18 | T={MC_T}",
    )

    return results, y_val


# ---------------------------------------------------------------------------
# Stage 4 — Deep Ensembles
# ---------------------------------------------------------------------------

def stage_ensemble():
    print("\n" + "=" * 60)
    print(f"STAGE: DEEP ENSEMBLES ({N_ENSEMBLE} members, volunteer-level split)")
    print("=" * 60)

    from models.deep_ensemble import (
        train_ensemble, ensemble_predict,
        save_members, save_learning_curves, save_uncertainty_plots, save_metrics,
    )

    X, y, meta = _load_data()
    target_names = TARGET_COLS[:y.shape[1]]

    X_train, X_val, y_train, y_val, _, _ = volunteer_split(X, y, meta)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)

    input_dim = X_train_s.shape[1]
    n_targets = y_train.shape[1]

    members, histories = train_ensemble(
        X_train_s, y_train, X_val_s, y_val,
        n_members=N_ENSEMBLE,
        input_dim=input_dim,
        n_targets=n_targets,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=PATIENCE,
    )

    print("\n[ensemble] Running inference ...")
    results = ensemble_predict(members, X_val_s)

    print("\n[ensemble] Saving outputs ...")
    save_members(members, run_tag="ensemble")
    save_learning_curves(histories, run_tag="ensemble")
    save_uncertainty_plots(results, y_val, target_names, run_tag="ensemble")
    save_metrics(
        results, y_val, target_names, run_tag="ensemble",
        extra_info=f"Train vols: 1-14 | Val vols: 15-18 | Members={N_ENSEMBLE}",
    )

    return results, y_val


# ---------------------------------------------------------------------------
# Stage 5 — March 2023 Independent Validation
# ---------------------------------------------------------------------------

def stage_validate():
    """
    Phase 4: Evaluate all trained models on the independent March 2023 dataset
    (24 new volunteers, ultrasound-verified labels — never seen during training).

    Steps:
      1. Load March 2023 features + labels via load_validation_data()
      2. Apply the September 2022 StandardScaler (must NOT refit — that would
         leak test statistics into the preprocessing step)
      3. Load the latest saved MC Dropout model → run T=50 stochastic passes
      4. Load the latest saved Ensemble members → aggregate predictions
      5. Retrain baselines (RF, XGBoost, FCNN) on the FULL September 2022 data
         (all 431 samples — no held-out split needed since March 2023 is the
         independent test set), then predict on March 2023
      6. Compute RMSE, MAE, R², ECE, PICP, MPIW, NLL for all models × targets
      7. Generate reliability diagrams and scatter plots
      8. Write a final comparison text report
    """
    print("\n" + "=" * 70)
    print("STAGE: INDEPENDENT VALIDATION — March 2023 (24 new volunteers)")
    print("=" * 70)

    from models.mc_dropout import (
        MCDropoutRegressor, load_model as load_mc_model,
        mc_predict, save_uncertainty_plots,
    )
    from models.deep_ensemble import (
        EnsembleMember, load_members, ensemble_predict,
    )
    from models.baselines import get_rf, get_xgb, get_fcnn, train_eval_baseline
    from evaluation.metrics import full_uncertainty_report
    from evaluation.plots import (
        plot_predictions_vs_truth, plot_reliability_diagram, save_fig,
    )
    import torch
    from config import MODELS_DIR

    # ------------------------------------------------------------------
    # 1. Load March 2023 validation data
    # ------------------------------------------------------------------
    print("\n[validate] Loading March 2023 independent validation data ...")
    X_test, y_test, meta_test = load_validation_data()
    target_names = TARGET_COLS[:y_test.shape[1]]
    n_test = X_test.shape[0]
    print(f"[validate] Test set: {n_test} samples, {meta_test['volunteer_id'].nunique()} volunteers")

    # ------------------------------------------------------------------
    # 2. Load the saved scaler (fitted on September 2022 training data)
    #    Apply WITHOUT refitting — preserves the original feature scale
    # ------------------------------------------------------------------
    scaler_path = OUTPUT_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            "outputs/scaler.pkl not found. Run --stage baseline or --stage mc_dropout first "
            "to fit and cache the StandardScaler on September 2022 data."
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[validate] Loaded scaler from {scaler_path}")
    X_test_s = scaler.transform(X_test).astype(np.float32)
    X_test_b  = add_band_features(X_test)   # band features for tree models

    # ------------------------------------------------------------------
    # 3. Load the latest MC Dropout model
    # ------------------------------------------------------------------
    mc_files = sorted(MODELS_DIR.glob("mc_dropout_*.pt"))
    if not mc_files:
        raise FileNotFoundError("No mc_dropout_*.pt files found in outputs/models/. Run --stage mc_dropout first.")
    mc_path = mc_files[-1]
    print(f"\n[validate] Loading MC Dropout model: {mc_path.name}")
    mc_model = load_mc_model(mc_path, input_dim=800, hidden_dims=(256, 128, 64), n_targets=3)

    print(f"[validate] Running MC Dropout inference (T={MC_T}) ...")
    mc_results = mc_predict(mc_model, X_test_s, T=MC_T)

    # ------------------------------------------------------------------
    # 4. Load the latest Ensemble members
    # ------------------------------------------------------------------
    # Members share a timestamp — find the latest set
    all_member_files = sorted(MODELS_DIR.glob("ensemble_member*_*.pt"))
    if not all_member_files:
        raise FileNotFoundError("No ensemble member files found. Run --stage ensemble first.")
    # Group by timestamp (last part before .pt)
    from collections import defaultdict
    ts_groups = defaultdict(list)
    for f in all_member_files:
        ts = "_".join(f.stem.split("_")[-2:])   # e.g. "20260415_204209"
        ts_groups[ts].append(f)
    latest_ts = sorted(ts_groups.keys())[-1]
    member_paths = ts_groups[latest_ts]
    print(f"\n[validate] Loading {len(member_paths)} ensemble members (ts={latest_ts}) ...")
    members = load_members(member_paths, input_dim=800, hidden_dims=(256, 128, 64), n_targets=3)

    print("[validate] Running ensemble inference ...")
    ens_results = ensemble_predict(members, X_test_s)

    # ------------------------------------------------------------------
    # 5. Retrain baselines on FULL September 2022 data, predict on March 2023
    # ------------------------------------------------------------------
    print("\n[validate] Retraining baselines on full September 2022 data ...")
    X_train, y_train, meta_train = _load_data()

    scaler_full = StandardScaler()
    X_train_s_full = scaler_full.fit_transform(X_train).astype(np.float32)
    X_test_s_full  = scaler_full.transform(X_test).astype(np.float32)
    X_train_b_full = add_band_features(X_train)

    baseline_models = {
        "Random Forest": (get_rf(),  X_train_b_full, X_test_b),
        "XGBoost":       (get_xgb(), X_train_b_full, X_test_b),
        "FCNN":          (get_fcnn(), X_train_s_full, X_test_s_full),
    }
    baseline_results = {}
    for name, (model, Xtr, Xte) in baseline_models.items():
        res = train_eval_baseline(name, model, Xtr, Xte, y_train, y_test, target_names)
        baseline_results[name] = res
        rmse_vals = res["metrics"]["RMSE"].tolist()
        print(f"[validate]   {name}: RMSE = {[f'{r:.3f}' for r in rmse_vals]}")

    # ------------------------------------------------------------------
    # 6. Compute all metrics and generate plots
    # ------------------------------------------------------------------
    print("\n[validate] Computing metrics and generating plots ...")

    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Collect all model predictions for the summary report
    all_model_metrics = {}   # model_name → {target → Series}

    units_map = {"Skin_mm": "mm", "Fat_mm": "mm", "Muscle_cm2": "cm²"}

    # --- MC Dropout ---
    print("[validate] Generating MC Dropout plots ...")
    mc_metrics = {}
    for i, tname in enumerate(target_names):
        y_true  = y_test[:, i]
        y_mean  = mc_results[f"mean_{tname}"]
        y_std   = mc_results[f"total_std_{tname}"]

        report = full_uncertainty_report(y_true, y_mean, y_std, target_name=tname)
        mc_metrics[tname] = report

        fig = plot_predictions_vs_truth(y_true, y_mean, y_std,
                                        target_name=tname, units=units_map[tname])
        save_fig(fig, f"val_mc_dropout_{tname}_{ts_now}")

        fig = plot_reliability_diagram(y_true, y_mean, y_std, target_name=f"MC Dropout — {tname}")
        save_fig(fig, f"val_reliability_mc_dropout_{tname}_{ts_now}")
    all_model_metrics["MC Dropout"] = mc_metrics

    # --- Deep Ensemble ---
    print("[validate] Generating Deep Ensemble plots ...")
    ens_metrics = {}
    for i, tname in enumerate(target_names):
        y_true  = y_test[:, i]
        y_mean  = ens_results[f"mean_{tname}"]
        y_std   = ens_results[f"total_std_{tname}"]

        report = full_uncertainty_report(y_true, y_mean, y_std, target_name=tname)
        ens_metrics[tname] = report

        fig = plot_predictions_vs_truth(y_true, y_mean, y_std,
                                        target_name=tname, units=units_map[tname])
        save_fig(fig, f"val_ensemble_{tname}_{ts_now}")

        fig = plot_reliability_diagram(y_true, y_mean, y_std, target_name=f"Ensemble — {tname}")
        save_fig(fig, f"val_reliability_ensemble_{tname}_{ts_now}")
    all_model_metrics["Deep Ensemble"] = ens_metrics

    # --- Baselines (no uncertainty, use zero std as placeholder) ---
    print("[validate] Generating baseline plots ...")
    for bname, bres in baseline_results.items():
        bm = {}
        for i, tname in enumerate(target_names):
            y_true = y_test[:, i]
            y_mean = bres["y_pred"][:, i]
            # Baselines are deterministic: std = 0 (no intervals); skip calibration plots
            report = full_uncertainty_report(
                y_true, y_mean, np.zeros_like(y_mean), target_name=tname
            )
            bm[tname] = report

            fig = plot_predictions_vs_truth(y_true, y_mean, target_name=tname,
                                            units=units_map[tname])
            save_fig(fig, f"val_{bname.lower().replace(' ', '_')}_{tname}_{ts_now}")
        all_model_metrics[bname] = bm

    # ------------------------------------------------------------------
    # 7. Write the final comparison report
    # ------------------------------------------------------------------
    _save_validation_report(all_model_metrics, target_names, y_test,
                             mc_results, ens_results, ts_now)

    print("\n[validate] Phase 4 complete.")
    print(f"[validate] All figures in outputs/figures/")
    print(f"[validate] Report in outputs/metrics/")

    return all_model_metrics


def _save_validation_report(
    all_model_metrics: dict,
    target_names: list,
    y_test: np.ndarray,
    mc_results: dict,
    ens_results: dict,
    ts: str,
) -> None:
    """Write a formatted text comparison report for the March 2023 validation."""
    from config import METRICS_DIR

    lines = [
        "=" * 78,
        "INDEPENDENT VALIDATION REPORT — March 2023 Dataset (24 volunteers)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "All models evaluated on unseen volunteers (zero overlap with September 2022 training).",
        "=" * 78,
        "",
    ]

    metric_keys = ["RMSE", "MAE", "R2", "PICP_95", "MPIW_95", "ECE", "NLL"]
    metric_labels = {
        "RMSE":    "RMSE",
        "MAE":     "MAE",
        "R2":      "R²",
        "PICP_95": "PICP 95%",
        "MPIW_95": "MPIW 95%",
        "ECE":     "ECE",
        "NLL":     "NLL",
    }

    model_names = list(all_model_metrics.keys())

    for tname in target_names:
        lines.append(f"TARGET: {tname}")
        lines.append("-" * 78)
        header = f"  {'Metric':<14}" + "".join(f"{m:<20}" for m in model_names)
        lines.append(header)
        lines.append("  " + "-" * (14 + 20 * len(model_names)))

        for key in metric_keys:
            label = metric_labels[key]
            row = f"  {label:<14}"
            for mname in model_names:
                series = all_model_metrics[mname][tname]
                val = series.get(key, float("nan"))
                row += f"{val:<20.4f}"
            lines.append(row)

        # Uncertainty decomposition for probabilistic models
        lines.append("")
        lines.append("  Uncertainty decomposition (probabilistic models only):")
        for mname, results_dict in [("MC Dropout", mc_results), ("Deep Ensemble", ens_results)]:
            epi = results_dict.get(f"epistemic_{tname}", np.array([np.nan]))
            ale = results_dict.get(f"aleatoric_{tname}", np.array([np.nan]))
            lines.append(f"    {mname:<20} epistemic σ={np.mean(epi):.4f}  aleatoric σ={np.mean(ale):.4f}")
        lines.append("")

    # Best model summary
    lines += [
        "=" * 78,
        "BEST MODEL PER TARGET (by RMSE on March 2023):",
        "-" * 78,
    ]
    for tname in target_names:
        best_name, best_rmse = None, float("inf")
        for mname in model_names:
            rmse_val = all_model_metrics[mname][tname].get("RMSE", float("inf"))
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_name = mname
        lines.append(f"  {tname:<20} → {best_name:<25} RMSE={best_rmse:.4f}")

    lines += ["", "=" * 78, "END OF REPORT", "=" * 78]

    path = METRICS_DIR / f"validation_march2023_{ts}.txt"
    path.write_text("\n".join(lines))
    print(f"[validate] Report saved → {path}")


# ---------------------------------------------------------------------------
# Stage 6 — Clinical Risk Score
# ---------------------------------------------------------------------------

def stage_risk_score():
    """
    Phase 5: Clinical risk stratification using predictive uncertainty.

    For each prediction, σ_total tells us how confident the model is.
    We split predictions into:
      - Low-risk  (σ_total ≤ threshold): model is confident — lower RMSE
      - High-risk (σ_total >  threshold): model is uncertain — higher RMSE

    We use the 75th percentile of σ_total as the threshold (flags top 25%
    most uncertain predictions). We also sweep across flagging rates 5%–60%
    to show how discrimination power evolves.

    Only the two probabilistic models (MC Dropout, Deep Ensemble) produce
    σ_total — baselines have no uncertainty estimates so cannot be risk-stratified.
    """
    print("\n" + "=" * 70)
    print("STAGE: CLINICAL RISK SCORE")
    print("=" * 70)

    from models.mc_dropout import MCDropoutRegressor, load_model as load_mc_model, mc_predict
    from models.deep_ensemble import EnsembleMember, load_members, ensemble_predict
    from evaluation.risk_score import generate_risk_report
    import torch
    from config import MODELS_DIR

    # ------------------------------------------------------------------
    # 1. Load March 2023 data and scale it
    # ------------------------------------------------------------------
    print("\n[risk] Loading March 2023 validation data ...")
    X_test, y_test, meta_test = load_validation_data()
    target_names = TARGET_COLS[:y_test.shape[1]]

    scaler_path = OUTPUT_DIR / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    print(f"[risk] Test set: {X_test.shape[0]} samples, {meta_test['volunteer_id'].nunique()} volunteers")

    # ------------------------------------------------------------------
    # 2. MC Dropout inference
    # ------------------------------------------------------------------
    mc_files = sorted(MODELS_DIR.glob("mc_dropout_*.pt"))
    if not mc_files:
        raise FileNotFoundError("No mc_dropout_*.pt found. Run --stage mc_dropout first.")
    print(f"\n[risk] Loading MC Dropout: {mc_files[-1].name}")
    mc_model = load_mc_model(mc_files[-1], input_dim=800, hidden_dims=(256, 128, 64), n_targets=3)
    mc_results = mc_predict(mc_model, X_test_s, T=MC_T)

    # ------------------------------------------------------------------
    # 3. Ensemble inference
    # ------------------------------------------------------------------
    all_member_files = sorted(MODELS_DIR.glob("ensemble_member*_*.pt"))
    if not all_member_files:
        raise FileNotFoundError("No ensemble members found. Run --stage ensemble first.")
    from collections import defaultdict
    ts_groups = defaultdict(list)
    for f in all_member_files:
        ts = "_".join(f.stem.split("_")[-2:])
        ts_groups[ts].append(f)
    member_paths = ts_groups[sorted(ts_groups.keys())[-1]]
    print(f"[risk] Loading {len(member_paths)} ensemble members ...")
    members = load_members(member_paths, input_dim=800, hidden_dims=(256, 128, 64), n_targets=3)
    ens_results = ensemble_predict(members, X_test_s)

    # ------------------------------------------------------------------
    # 4. Build the all_results dict expected by generate_risk_report
    #    Format: model_name → {target_name → (y_true, y_mean, total_std)}
    # ------------------------------------------------------------------
    all_results = {}

    mc_targets = {}
    for i, tname in enumerate(target_names):
        mc_targets[tname] = (
            y_test[:, i],
            mc_results[f"mean_{tname}"],
            mc_results[f"total_std_{tname}"],
        )
    all_results["MC Dropout"] = mc_targets

    ens_targets = {}
    for i, tname in enumerate(target_names):
        ens_targets[tname] = (
            y_test[:, i],
            ens_results[f"mean_{tname}"],
            ens_results[f"total_std_{tname}"],
        )
    all_results["Deep Ensemble"] = ens_targets

    # ------------------------------------------------------------------
    # 5. Generate full risk report + plots
    # ------------------------------------------------------------------
    print("\n[risk] Running risk stratification analysis ...")
    report_path = generate_risk_report(all_results, target_names, run_tag="risk_score")

    print(f"\n[risk] Phase 5 complete.")
    print(f"[risk] Report: {report_path.name}")
    print(f"[risk] Figures in outputs/figures/")

    return all_results


# ---------------------------------------------------------------------------
# Stage 7 — Comparison Table
# ---------------------------------------------------------------------------

def stage_compare():
    """
    Print a side-by-side comparison table of all saved metrics files.
    """
    print("\n" + "=" * 60)
    print("STAGE: MODEL COMPARISON")
    print("=" * 60)

    from config import METRICS_DIR
    metrics_files = sorted(METRICS_DIR.glob("*.txt"))
    if not metrics_files:
        print("[compare] No metrics files found in outputs/metrics/. Run other stages first.")
        return

    print("\nMetrics files found:")
    for f in metrics_files:
        print(f"  {f.name}")

    print("\nFor a full comparison table, open the metrics files above.")
    print("A formatted comparison plot will be added in the calibration analysis stage.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Thesis ML Pipeline")
    parser.add_argument(
        "--stage",
        choices=["data", "baseline", "cv", "mc_dropout", "ensemble", "validate", "risk_score", "compare", "all"],
        default="data",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit S2P files loaded (for quick testing)")
    args = parser.parse_args()

    if args.stage == "data" or args.stage == "all":
        stage_data(limit=args.limit)

    if args.stage in ("baseline", "cv", "all"):
        stage_cv()

    if args.stage == "mc_dropout" or args.stage == "all":
        stage_mc_dropout()

    if args.stage == "ensemble" or args.stage == "all":
        stage_ensemble()

    if args.stage == "validate" or args.stage == "all":
        stage_validate()

    if args.stage == "risk_score" or args.stage == "all":
        stage_risk_score()

    if args.stage == "compare" or args.stage == "all":
        stage_compare()

    print("\n[pipeline] Done.")


if __name__ == "__main__":
    main()
