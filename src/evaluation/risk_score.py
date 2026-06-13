"""
risk_score.py
=============
Clinical risk stratification based on predictive uncertainty.

Core idea (thesis Chapter 5):
  A calibrated uncertainty model doesn't just predict a value — it also knows
  WHEN it is likely to be wrong. This module operationalises that property:

  1. Compute total predictive std (σ_total = √(σ_epistemic² + σ_aleatoric²))
     for every test sample.
  2. Apply a threshold T: samples with σ_total > T are "high-risk" (uncertain);
     samples with σ_total ≤ T are "low-risk" (confident).
  3. Measure RMSE on each stratum. A useful uncertainty model produces
     lower RMSE on the low-risk stratum — it can self-identify reliable predictions.

  This is directly clinically relevant: a clinician using this system could
  be alerted "this measurement is uncertain — consider ultrasound confirmation"
  when σ_total exceeds the threshold.

Threshold selection:
  Rather than a single ad-hoc threshold, we sweep across quantiles of σ_total
  (e.g. flag top 10%, 20%, 30%, 40%, 50% most uncertain predictions). For each
  flagging rate we report:
    - RMSE on low-risk subset (confident predictions — what the model is sure about)
    - RMSE on high-risk subset (uncertain predictions — where errors are expected)
    - Separation ratio = RMSE_high / RMSE_low  (higher = better risk discrimination)

  The "recommended threshold" is chosen where the low-risk RMSE drops most
  steeply (largest marginal improvement), typically around the 25–40th percentile.

Functions:
  compute_risk_flags()          — boolean mask: True = high-risk
  risk_stratified_metrics()     — RMSE/MAE/coverage per stratum
  sweep_thresholds()            — sweep flagging rates, collect metrics
  find_recommended_threshold()  — find threshold with best discrimination
  plot_risk_stratification()    — 3-panel visualisation
  plot_threshold_sweep()        — RMSE vs flagging-rate curve
  generate_risk_report()        — write full text report
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, METRICS_DIR, TARGET_COLS
from evaluation.plot_style import apply_pub_style, PALETTE, TARGET_UNITS, TARGET_DISPLAY
apply_pub_style()


# ---------------------------------------------------------------------------
# 1. Core risk flagging
# ---------------------------------------------------------------------------

def compute_risk_flags(
    total_std:       np.ndarray,
    threshold:       float,
) -> np.ndarray:
    """
    Return a boolean mask: True where σ_total > threshold (high-risk / uncertain).

    Args:
        total_std : (n,) array of per-sample total predictive std
        threshold : scalar threshold on σ_total

    Returns:
        high_risk : (n,) boolean array
    """
    return total_std > threshold


def risk_stratified_metrics(
    y_true:    np.ndarray,
    y_mean:    np.ndarray,
    total_std: np.ndarray,
    threshold: float,
    coverage:  float = 0.95,
) -> Dict:
    """
    Compute metrics for the low-risk and high-risk strata separately.

    Args:
        y_true     : (n,) ground truth
        y_mean     : (n,) predictive mean
        total_std  : (n,) total predictive std
        threshold  : σ threshold separating low-risk from high-risk
        coverage   : nominal confidence level (default 0.95)

    Returns:
        dict with keys:
          n_low, n_high, frac_flagged
          rmse_low, rmse_high, mae_low, mae_high
          picp_low, picp_high
          separation_ratio  (RMSE_high / RMSE_low — higher = better discrimination)
    """
    from scipy import stats

    high_risk = compute_risk_flags(total_std, threshold)
    low_risk  = ~high_risk

    n_low  = int(low_risk.sum())
    n_high = int(high_risk.sum())
    n_total = len(y_true)

    # Avoid division by zero if one stratum is empty
    def _rmse(mask):
        if mask.sum() == 0:
            return np.nan
        return float(np.sqrt(np.mean((y_true[mask] - y_mean[mask]) ** 2)))

    def _mae(mask):
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(np.abs(y_true[mask] - y_mean[mask])))

    def _picp(mask):
        if mask.sum() == 0:
            return np.nan
        z = stats.norm.ppf((1 + coverage) / 2)
        lo = y_mean[mask] - z * total_std[mask]
        hi = y_mean[mask] + z * total_std[mask]
        return float(np.mean((y_true[mask] >= lo) & (y_true[mask] <= hi)))

    rmse_low  = _rmse(low_risk)
    rmse_high = _rmse(high_risk)
    sep_ratio = (rmse_high / rmse_low) if (rmse_low > 0 and not np.isnan(rmse_high)) else np.nan

    return {
        "threshold":       threshold,
        "n_low":           n_low,
        "n_high":          n_high,
        "frac_flagged":    n_high / n_total,
        "rmse_low":        rmse_low,
        "rmse_high":       rmse_high,
        "mae_low":         _mae(low_risk),
        "mae_high":        _mae(high_risk),
        "picp_low":        _picp(low_risk),
        "picp_high":       _picp(high_risk),
        "separation_ratio": sep_ratio,
    }


# ---------------------------------------------------------------------------
# 2. Threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    y_true:       np.ndarray,
    y_mean:       np.ndarray,
    total_std:    np.ndarray,
    flagging_rates: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Sweep thresholds corresponding to flagging rates 5%–60%.

    For each flagging rate f, the threshold is the (1-f) quantile of total_std.
    At flagging rate 0% → nobody flagged (all low-risk).
    At flagging rate 100% → everyone flagged.

    Returns a DataFrame with one row per threshold.
    """
    if flagging_rates is None:
        flagging_rates = np.arange(0.05, 0.65, 0.05)

    rows = []
    for frac in flagging_rates:
        threshold = float(np.quantile(total_std, 1.0 - frac))
        row = risk_stratified_metrics(y_true, y_mean, total_std, threshold)
        row["flagging_rate_nominal"] = frac
        rows.append(row)

    return pd.DataFrame(rows)


def find_recommended_threshold(
    sweep_df: pd.DataFrame,
    min_flagging_rate: float = 0.10,
    max_flagging_rate: float = 0.50,
) -> Dict:
    """
    Select the threshold with the highest separation ratio
    within the useful flagging-rate window.

    A separation ratio > 1.0 means the high-risk stratum has worse RMSE
    than the low-risk stratum — the uncertainty estimate is informative.
    """
    mask = (
        (sweep_df["flagging_rate_nominal"] >= min_flagging_rate) &
        (sweep_df["flagging_rate_nominal"] <= max_flagging_rate) &
        (sweep_df["separation_ratio"].notna())
    )
    sub = sweep_df[mask]
    if sub.empty:
        return sweep_df.iloc[len(sweep_df) // 2].to_dict()

    best_idx = sub["separation_ratio"].idxmax()
    return sweep_df.loc[best_idx].to_dict()


# ---------------------------------------------------------------------------
# 3. Visualisation
# ---------------------------------------------------------------------------

def plot_risk_stratification(
    y_true:      np.ndarray,
    y_mean:      np.ndarray,
    total_std:   np.ndarray,
    threshold:   float,
    target_name: str = "Target",
    units:       str = "",
) -> plt.Figure:
    """
    Three-panel clinical risk stratification figure.

    Panel 1 — Scatter: predicted vs ground truth, coloured by risk tier.
      Blue = low-risk (model is confident, σ_total ≤ threshold).
      Red  = high-risk (model is uncertain, σ_total > threshold).
      Low-risk points should cluster more tightly around the diagonal,
      visually confirming that confidence correlates with accuracy.

    Panel 2 — Absolute prediction error vs σ_total.
      A positive slope in the trend line confirms that σ_total is a
      reliable proxy for actual prediction quality — the core requirement
      for a clinically useful risk flag.  The vertical dashed line marks
      the threshold.

    Panel 3 — RMSE bar chart: low-risk vs high-risk stratum.
      The separation ratio ρ = RMSE_high / RMSE_low quantifies how well
      the model discriminates reliable from unreliable predictions.
      ρ > 1.0 means high-uncertainty predictions are genuinely less accurate.

    Parameters
    ----------
    y_true      : Ground-truth tissue measurements.
    y_mean      : Model predicted means.
    total_std   : Total predictive standard deviation per sample.
    threshold   : Uncertainty threshold (75th percentile of σ_total).
    target_name : Target variable name for labels.
    units       : Physical unit string.
    """
    disp      = TARGET_DISPLAY.get(target_name, target_name)
    unit_s    = units or TARGET_UNITS.get(target_name, "")
    lbl       = f"({unit_s})" if unit_s else ""

    high_risk = compute_risk_flags(total_std, threshold)
    low_risk  = ~high_risk
    errors    = np.abs(y_true - y_mean)

    rmse_low  = float(np.sqrt(np.mean(errors[low_risk] ** 2)))  if low_risk.sum()  > 0 else np.nan
    rmse_high = float(np.sqrt(np.mean(errors[high_risk] ** 2))) if high_risk.sum() > 0 else np.nan
    sep_ratio = rmse_high / rmse_low if (not np.isnan(rmse_low) and rmse_low > 0) else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Clinical Risk Stratification — {disp}\n"
        f"Threshold: σ_total = {threshold:.3f} {unit_s}  |  "
        f"Separation ratio ρ = {sep_ratio:.3f}"
    )

    # ── Panel 1: Scatter coloured by risk tier ──────────────────────────────
    ax = axes[0]
    lims = [
        min(y_true.min(), y_mean.min()) * 0.96,
        max(y_true.max(), y_mean.max()) * 1.04,
    ]
    if low_risk.sum() > 0:
        ax.scatter(y_true[low_risk], y_mean[low_risk], s=45, alpha=0.72,
                   color=PALETTE["low_risk"], edgecolors="white", linewidths=0.4,
                   label=f"Low-risk  (n={low_risk.sum()},  σ ≤ {threshold:.3f})")
    if high_risk.sum() > 0:
        ax.scatter(y_true[high_risk], y_mean[high_risk], s=45, alpha=0.72,
                   color=PALETTE["high_risk"], edgecolors="white", linewidths=0.4,
                   label=f"High-risk (n={high_risk.sum()},  σ > {threshold:.3f})")
    ax.plot(lims, lims, "--", color=PALETTE["perfect"],
            linewidth=2.0, label="Perfect prediction", zorder=0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"Ground Truth — {disp} {lbl}")
    ax.set_ylabel(f"Predicted — {disp} {lbl}")
    ax.set_title("Predicted vs. Ground Truth\nby Risk Tier")
    ax.legend(fontsize=12)

    # ── Panel 2: |Error| vs σ_total ─────────────────────────────────────────
    ax = axes[1]
    if low_risk.sum() > 0:
        ax.scatter(total_std[low_risk], errors[low_risk], s=40, alpha=0.65,
                   color=PALETTE["low_risk"], edgecolors="white", linewidths=0.4,
                   label="Low-risk")
    if high_risk.sum() > 0:
        ax.scatter(total_std[high_risk], errors[high_risk], s=40, alpha=0.65,
                   color=PALETTE["high_risk"], edgecolors="white", linewidths=0.4,
                   label="High-risk")
    ax.axvline(threshold, color=PALETTE["threshold"], linestyle="--",
               linewidth=2.2, label=f"Threshold = {threshold:.3f} {unit_s}")
    if len(total_std) > 5:
        z    = np.polyfit(total_std, errors, 1)
        xfit = np.linspace(total_std.min(), total_std.max(), 200)
        ax.plot(xfit, np.polyval(z, xfit), color=PALETTE["trend"],
                linewidth=2.5, alpha=0.85, label="Linear trend")
    ax.set_xlabel(f"Total Predictive σ {lbl}")
    ax.set_ylabel(f"Absolute Prediction Error {lbl}")
    ax.set_title("|Prediction Error| vs. σ_total\n(slope > 0 validates risk flag)")
    ax.legend(fontsize=12)

    # ── Panel 3: RMSE bar chart ─────────────────────────────────────────────
    ax = axes[2]
    strata, vals, bar_colors = [], [], []
    if not np.isnan(rmse_low):
        strata.append(f"Low-risk\n(n = {low_risk.sum()})")
        vals.append(rmse_low)
        bar_colors.append(PALETTE["low_risk"])
    if not np.isnan(rmse_high):
        strata.append(f"High-risk\n(n = {high_risk.sum()})")
        vals.append(rmse_high)
        bar_colors.append(PALETTE["high_risk"])

    bars = ax.bar(strata, vals, color=bar_colors, alpha=0.88,
                  edgecolor="white", linewidth=1.5, width=0.55)
    y_max = max(vals) if vals else 1.0
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.015,
                f"{val:.3f} {unit_s}",
                ha="center", va="bottom", fontsize=14, fontweight="bold")
    if not np.isnan(sep_ratio):
        ax.text(0.97, 0.97, f"ρ = {sep_ratio:.3f}×",
                transform=ax.transAxes, va="top", ha="right", fontsize=14,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.9))
    ax.set_ylabel(f"RMSE {lbl}")
    ax.set_title("RMSE by Risk Stratum\n(lower is better for low-risk)")
    ax.set_ylim(0, y_max * 1.20)

    return fig


def plot_threshold_sweep(
    sweep_df:    pd.DataFrame,
    target_name: str = "Target",
) -> plt.Figure:
    """
    Two-panel threshold sensitivity analysis figure.

    As the flagging rate increases (more predictions sent for follow-up),
    the low-risk pool shrinks but becomes more accurate; the high-risk pool
    grows and accumulates harder cases.

    Panel 1 — RMSE per stratum vs flagging rate:
      Low-risk RMSE (blue) should decrease as flagging rate rises, because
      more uncertain samples are removed from the accepted pool.
      High-risk RMSE (red) should stay equal or rise.
      The gap between the two curves quantifies the practical benefit of
      applying the risk flag at each operating point.

    Panel 2 — Separation ratio ρ = RMSE_high / RMSE_low:
      A ratio of 1.0 means the model cannot distinguish reliable from
      unreliable predictions.  Any value above 1.0 confirms genuine
      discrimination.  The optimal flagging rate (recommended threshold)
      corresponds to the peak separation ratio.

    Parameters
    ----------
    sweep_df    : DataFrame with columns flagging_rate_nominal, rmse_low,
                  rmse_high, separation_ratio.
    target_name : Target variable name for title.
    """
    disp = TARGET_DISPLAY.get(target_name, target_name)
    fr   = sweep_df["flagging_rate_nominal"] * 100   # as percentage

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Uncertainty Threshold Sweep — {disp}")

    # ── Panel 1: RMSE per stratum vs flagging rate ──────────────────────────
    ax = axes[0]
    ax.plot(fr, sweep_df["rmse_low"],  "o-", color=PALETTE["low_risk"],
            linewidth=2.5, markersize=8, label="Low-risk RMSE (accepted)")
    ax.plot(fr, sweep_df["rmse_high"], "s-", color=PALETTE["high_risk"],
            linewidth=2.5, markersize=8, label="High-risk RMSE (flagged)")
    ax.fill_between(fr, sweep_df["rmse_low"], sweep_df["rmse_high"],
                    alpha=0.12, color=PALETTE["low_risk"])
    ax.set_xlabel("Flagging Rate — fraction of samples sent for follow-up (%)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by Stratum\nvs Flagging Rate")
    ax.legend()

    # ── Panel 2: Separation ratio ────────────────────────────────────────────
    ax = axes[1]
    ax.plot(fr, sweep_df["separation_ratio"], "D-",
            color=PALETTE["ensemble"], linewidth=2.5, markersize=8,
            label="Separation ratio ρ")
    ax.axhline(1.0, color=PALETTE["perfect"], linestyle="--",
               linewidth=2.0, label="ρ = 1.0  (no discrimination)")
    ax.fill_between(fr, 1.0, sweep_df["separation_ratio"],
                    where=(sweep_df["separation_ratio"] >= 1.0),
                    alpha=0.15, color=PALETTE["ensemble"],
                    label="Gain over random flagging")
    ax.set_xlabel("Flagging Rate (%)")
    ax.set_ylabel("Separation Ratio  ρ = RMSE_high / RMSE_low")
    ax.set_title("Uncertainty Discrimination Power\n(ρ > 1.0 confirms valid risk flag)")
    ax.legend()

    return fig


# ---------------------------------------------------------------------------
# 4. Full report generation
# ---------------------------------------------------------------------------

def generate_risk_report(
    all_results:   Dict,       # model_name → {target → (y_true, y_mean, total_std)}
    target_names:  List[str],
    run_tag:       str = "risk_score",
) -> Path:
    """
    Run the full risk stratification analysis for each model × target,
    generate all plots, and write a text report.

    Args:
        all_results  : dict mapping model_name → dict mapping target_name →
                       (y_true, y_mean, total_std) as np.ndarray triples
        target_names : list of target column names
        run_tag      : prefix for saved files

    Returns:
        Path to the saved text report.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    units_map = {"Skin_mm": "mm", "Fat_mm": "mm", "Muscle_cm2": "cm²"}

    report_lines = [
        "=" * 78,
        "CLINICAL RISK SCORE REPORT — Uncertainty-Based Prediction Flagging",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Dataset   : March 2023 (471 samples, 23 unseen volunteers)",
        "",
        "Strategy  : Predictions with σ_total above a threshold are flagged as",
        "            'high-risk' (model is uncertain). The threshold is chosen as",
        "            the 75th percentile of σ_total (flags top 25% most uncertain).",
        "            We verify that flagged predictions have higher RMSE — confirming",
        "            that σ_total is a reliable indicator of prediction quality.",
        "=" * 78,
        "",
    ]

    for model_name, target_dict in all_results.items():
        report_lines += [
            f"MODEL: {model_name}",
            "-" * 78,
        ]

        for tname in target_names:
            if tname not in target_dict:
                continue
            y_true, y_mean, total_std = target_dict[tname]
            units = units_map.get(tname, "")

            # Sweep thresholds
            sweep_df = sweep_thresholds(y_true, y_mean, total_std)
            rec      = find_recommended_threshold(sweep_df)

            # Use 25% flagging rate as the standard clinical threshold
            threshold = float(np.quantile(total_std, 0.75))
            metrics   = risk_stratified_metrics(y_true, y_mean, total_std, threshold)

            report_lines += [
                f"  Target: {tname}",
                f"    Recommended threshold  : σ_total > {threshold:.4f} {units}",
                f"    Samples flagged (25%)  : {metrics['n_high']} / {len(y_true)} "
                f"({metrics['frac_flagged']*100:.1f}%)",
                f"",
                f"    LOW-RISK  (confident)  : RMSE={metrics['rmse_low']:.4f}  "
                f"MAE={metrics['mae_low']:.4f}  PICP95={metrics['picp_low']:.3f}",
                f"    HIGH-RISK (uncertain)  : RMSE={metrics['rmse_high']:.4f}  "
                f"MAE={metrics['mae_high']:.4f}  PICP95={metrics['picp_high']:.3f}",
                f"    Separation ratio       : {metrics['separation_ratio']:.3f}x  "
                f"(RMSE_high / RMSE_low — higher = better discrimination)",
                f"",
            ]

            # Generate plots
            fig = plot_risk_stratification(
                y_true, y_mean, total_std, threshold,
                target_name=f"{model_name} — {tname}", units=units,
            )
            fig_path = FIGURES_DIR / f"{run_tag}_{model_name.lower().replace(' ', '_')}_{tname}_{ts}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[risk_score] Figure → {fig_path.name}")

            fig = plot_threshold_sweep(sweep_df, target_name=f"{model_name} — {tname}")
            sweep_path = FIGURES_DIR / f"{run_tag}_sweep_{model_name.lower().replace(' ', '_')}_{tname}_{ts}.png"
            fig.savefig(sweep_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[risk_score] Sweep plot → {sweep_path.name}")

        report_lines.append("")

    # Interpretation section
    report_lines += [
        "=" * 78,
        "INTERPRETATION",
        "-" * 78,
        "",
        "A separation ratio > 1.0 confirms that σ_total is informative: samples",
        "the model flagged as uncertain have genuinely higher prediction error.",
        "",
        "Clinical implication:",
        "  - For low-risk predictions (σ_total below threshold): the model's",
        "    output can be used with higher confidence as a screening result.",
        "  - For high-risk predictions (σ_total above threshold): the model is",
        "    signalling that the measurement may be unreliable. The clinician",
        "    should consider corroborating with ultrasound or another modality.",
        "",
        "This is the core contribution of uncertainty-aware modelling: not just",
        "a prediction, but a reliable self-assessment of prediction quality.",
        "",
        "=" * 78,
        "END OF REPORT",
        "=" * 78,
    ]

    path = METRICS_DIR / f"{run_tag}_{ts}.txt"
    path.write_text("\n".join(report_lines))
    print(f"[risk_score] Report saved → {path}")
    return path
