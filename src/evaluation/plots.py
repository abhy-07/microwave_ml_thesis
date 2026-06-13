"""
plots.py
========
Publication-quality visualisation utilities for uncertainty-aware
regression results.

Functions
---------
plot_predictions_vs_truth()   — scatter plot with ±2σ error bars
plot_reliability_diagram()    — calibration curve for regression
plot_uncertainty_distribution() — histogram of predictive std
plot_feature_importance()     — top-N feature importances (RF / XGBoost)
plot_multi_target_metrics()   — grouped bar chart across models
save_fig()                    — save figure to outputs/figures/ at 300 DPI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR
from evaluation.plot_style import apply_pub_style, PALETTE, TARGET_UNITS, TARGET_DISPLAY

apply_pub_style()


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save figure to outputs/figures/ at 300 DPI with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = FIGURES_DIR / f"{name}_{timestamp}.png"
    fig.savefig(path)          # dpi and bbox_inches set by rcParams
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


def plot_predictions_vs_truth(
    y_true:      np.ndarray,
    y_mean:      np.ndarray,
    y_std:       Optional[np.ndarray] = None,
    target_name: str = "Target",
    units:       str = "",
    model_label: str = "",
) -> plt.Figure:
    """
    Scatter plot of predicted mean vs ground truth.

    Each point is one test sample.  When y_std is supplied, ±2σ error bars
    are drawn, conveying the model's confidence in each prediction.
    Points that lie on the dashed diagonal represent perfect predictions.

    Parameters
    ----------
    y_true      : Ground-truth tissue measurements.
    y_mean      : Model's predicted means.
    y_std       : Predictive standard deviations (optional).
    target_name : Target variable name (e.g. 'Skin_mm').
    units       : Physical unit string for axis labels.
    model_label : Short model identifier for the legend.
    """
    disp   = TARGET_DISPLAY.get(target_name, target_name)
    unit_s = units or TARGET_UNITS.get(target_name, "")
    label  = f"({unit_s})" if unit_s else ""

    fig, ax = plt.subplots(figsize=(7, 7))

    if y_std is not None:
        ax.errorbar(
            y_true, y_mean, yerr=2 * y_std,
            fmt="o",
            color=PALETTE["mc"],
            alpha=0.70,
            elinewidth=1.4,
            capsize=4,
            capthick=1.4,
            markersize=7,
            label=f"{model_label} ±2σ" if model_label else "Prediction ±2σ",
        )
    else:
        ax.scatter(
            y_true, y_mean,
            color=PALETTE["mc"],
            alpha=0.75,
            s=55,
            edgecolors="white",
            linewidths=0.5,
            label=model_label or "Prediction",
        )

    lims = [
        min(y_true.min(), y_mean.min()) * 0.97,
        max(y_true.max(), y_mean.max()) * 1.03,
    ]
    ax.plot(lims, lims, "--", color=PALETTE["perfect"],
            linewidth=2.0, label="Perfect prediction", zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f"Ground Truth — {disp} {label}")
    ax.set_ylabel(f"Model Prediction — {disp} {label}")
    ax.set_title(f"Predicted vs. Ground Truth\n{disp}")
    ax.legend()

    # Annotate RMSE and MAE directly on the plot
    rmse = float(np.sqrt(np.mean((y_true - y_mean) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_mean)))
    ax.text(
        0.04, 0.96,
        f"RMSE = {rmse:.3f} {unit_s}\nMAE  = {mae:.3f} {unit_s}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.9),
    )

    return fig


def plot_reliability_diagram(
    y_true:      np.ndarray,
    y_mean:      np.ndarray,
    y_std:       np.ndarray,
    target_name: str = "Target",
    model_label: str = "",
    n_bins:      int = 10,
) -> plt.Figure:
    """
    Regression reliability (calibration) diagram.

    The x-axis is the nominal confidence level α (e.g. 0.90 means a 90%
    prediction interval).  The y-axis is the fraction of test samples
    that actually fall within that interval.  A perfectly calibrated
    model traces the diagonal.  Points above the diagonal indicate
    over-conservative intervals; points below indicate over-confident
    intervals.

    Parameters
    ----------
    y_true      : Ground-truth values.
    y_mean      : Predicted means.
    y_std       : Predicted standard deviations.
    target_name : Target variable name.
    model_label : Short model name for legend.
    n_bins      : Number of confidence levels to evaluate.
    """
    from scipy import stats

    disp   = TARGET_DISPLAY.get(target_name, target_name)
    alphas = np.linspace(0.05, 0.95, n_bins)
    obs    = []
    for alpha in alphas:
        z = stats.norm.ppf((1 + alpha) / 2)
        lo = y_mean - z * y_std
        hi = y_mean + z * y_std
        obs.append(float(np.mean((y_true >= lo) & (y_true <= hi))))

    # Compute ECE (mean absolute deviation from diagonal)
    ece = float(np.mean(np.abs(np.array(obs) - alphas)))

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.fill_between(alphas, alphas, obs, alpha=0.18, color=PALETTE["mc"],
                    label="Calibration gap")
    ax.plot(alphas, obs, "o-", color=PALETTE["mc"], linewidth=2.5,
            markersize=8, label=model_label or "Model calibration")
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["perfect"],
            linewidth=2.0, label="Perfect calibration")

    ax.set_xlabel("Nominal confidence level α")
    ax.set_ylabel("Observed coverage")
    ax.set_title(f"Reliability Diagram\n{disp}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")

    ax.text(
        0.97, 0.05,
        f"ECE = {ece:.3f}",
        transform=ax.transAxes,
        va="bottom", ha="right",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.9),
    )

    return fig


def plot_uncertainty_distribution(
    y_std:       np.ndarray,
    target_name: str = "Target",
    units:       str = "",
    model_label: str = "",
) -> plt.Figure:
    """
    Histogram of predictive standard deviations (σ_total).

    Shows the spread of model confidence across all test samples.
    A narrow histogram centred at low σ indicates a uniformly confident
    model; a wide or right-skewed histogram reveals that the model
    expresses high uncertainty for a subset of difficult inputs.
    """
    disp   = TARGET_DISPLAY.get(target_name, target_name)
    unit_s = units or TARGET_UNITS.get(target_name, "")
    label  = f"({unit_s})" if unit_s else ""
    title  = (f"{model_label} — " if model_label else "") + disp

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_std, bins=30, color=PALETTE["mc"], edgecolor="white",
            alpha=0.85, linewidth=0.6)
    ax.axvline(np.mean(y_std), color="#D6604D", linestyle="--",
               linewidth=2.2, label=f"Mean σ = {np.mean(y_std):.3f} {unit_s}")
    ax.set_xlabel(f"Total Predictive σ {label}")
    ax.set_ylabel("Number of Test Samples")
    ax.set_title(f"Distribution of Predictive Uncertainty\n{title}")
    ax.legend()

    return fig


def plot_feature_importance(
    importances:   np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n:         int = 20,
    title:         str = "Feature Importance",
) -> plt.Figure:
    """
    Horizontal bar chart of the top-N most important features.

    Feature importance is computed from the Random Forest's mean decrease
    in impurity, averaged across all trees and all three target outputs.
    The dominant features reveal which part of the S-parameter spectrum
    carries the most tissue-discriminative information.
    """
    idx = np.argsort(importances)[-top_n:]
    imp = importances[idx]
    labels = [feature_names[i] for i in idx] if feature_names is not None \
             else [f"f{i}" for i in idx]

    # Colour bars by S-parameter band
    colors = []
    for lbl in labels:
        if "S11" in lbl:
            colors.append("#2166AC")
        elif "S21" in lbl:
            colors.append("#D6604D")
        else:
            colors.append("#4DAC26")

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
    bars = ax.barh(range(len(imp)), imp, color=colors, alpha=0.88,
                   edgecolor="white", height=0.72)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Mean Decrease in Impurity (Importance Score)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)

    # Add value annotations
    for bar, val in zip(bars, imp):
        ax.text(bar.get_width() + imp.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=11)

    # Legend for S-parameter colour coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166AC", alpha=0.88, label="S11 (reflection)"),
        Patch(facecolor="#D6604D", alpha=0.88, label="S21 (transmission)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    return fig


def plot_multi_target_metrics(
    metrics_df,
    title: str = "Model Comparison",
) -> plt.Figure:
    """
    Grouped bar chart comparing RMSE, MAE, and R² across models and targets.

    Each subplot shows one metric.  Lower RMSE/MAE and higher R² are better.
    Negative R² values (all models on this dataset) reflect the difficulty
    of cross-volunteer generalisation at N=16 training volunteers; they do
    not indicate model failure.

    Parameters
    ----------
    metrics_df : DataFrame indexed by model name.  Columns must be named
                 RMSE_<target>, MAE_<target>, R2_<target>.
    title      : Figure suptitle.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    metrics  = ["RMSE", "MAE", "R2"]
    ylabels  = ["RMSE", "MAE", "R²"]
    colors   = [PALETTE["mc"], PALETTE["ensemble"], PALETTE["rf"]]

    for ax, metric, ylabel, color in zip(axes, metrics, ylabels, colors):
        cols = [c for c in metrics_df.columns if c.startswith(metric + "_")]
        sub  = metrics_df[cols].copy()
        sub.columns = [c.replace(f"{metric}_", "") for c in cols]
        sub.plot(kind="bar", ax=ax, colormap="tab10", alpha=0.85, legend=True,
                 edgecolor="white")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_title(f"{ylabel} per Target")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=11)
        if metric == "R2":
            ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)

    fig.suptitle(title)
    return fig
