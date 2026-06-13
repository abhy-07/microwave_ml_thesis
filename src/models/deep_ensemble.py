"""
deep_ensemble.py
================
Deep Ensembles for Uncertainty-Aware Regression.

Thesis: Uncertainty-Aware Machine Learning for Microwave-Based Body Composition Assessment

Theory (Lakshminarayanan et al., 2017):
  Train N independent neural networks with different random initialisations.
  Each network predicts (mean, log_variance) — i.e., is itself a probabilistic
  regressor (heteroscedastic). Ensemble uncertainty is computed from the
  mixture of Gaussians formed by the N members:

    μ* = (1/N) Σ μ_i
    σ*² = (1/N) Σ (σ_i² + μ_i²) − μ*²

  The total variance decomposes into:
    - Epistemic component: variance of the means across members
    - Aleatoric component: mean of the per-member predicted variances

  Deep Ensembles are considered the gold-standard non-Bayesian baseline
  for uncertainty quantification. They are simple, highly parallelisable,
  and empirically outperform many Bayesian approximations.

Architecture:
  Each ensemble member = same architecture as MCDropoutRegressor but
  WITHOUT Dropout (deterministic training, stochasticity comes from
  random initialisation + mini-batch ordering).

Reference:
  Lakshminarayanan et al. (2017). Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles. NeurIPS.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RANDOM_SEED, BATCH_SIZE, MAX_EPOCHS, LR, PATIENCE,
    N_ENSEMBLE, MODELS_DIR, FIGURES_DIR, METRICS_DIR, TARGET_COLS,
)
from evaluation.plot_style import apply_pub_style, PALETTE, TARGET_UNITS, TARGET_DISPLAY
apply_pub_style()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Single Ensemble Member (deterministic, heteroscedastic)
# ---------------------------------------------------------------------------

class EnsembleMember(nn.Module):
    """
    Deterministic probabilistic regressor — one member of a Deep Ensemble.

    No Dropout: stochasticity at the ensemble level comes from independent
    random weight initialisation and training data order (mini-batch SGD).

    Architecture: 800 → 256 → 128 → 64 → [mean, log_var] per target
    """

    def __init__(
        self,
        input_dim:   int = 800,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        n_targets:   int = 3,
    ) -> None:
        super().__init__()
        self.n_targets = n_targets

        encoder_layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            encoder_layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            ]
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.heads   = nn.ModuleList([nn.Linear(in_dim, 2) for _ in range(n_targets)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        stacked  = torch.stack([head(h) for head in self.heads], dim=1)  # (N, n_targets, 2)
        means    = stacked[:, :, 0]   # (N, n_targets)
        log_vars = stacked[:, :, 1]   # (N, n_targets)
        return means, log_vars


def _heteroscedastic_nll(
    means:    torch.Tensor,
    log_vars: torch.Tensor,
    targets:  torch.Tensor,
) -> torch.Tensor:
    log_vars = torch.clamp(log_vars, -10.0, 10.0)
    return (torch.exp(-log_vars) * (targets - means) ** 2 + log_vars).mean()


# ---------------------------------------------------------------------------
# 2. Train a Single Member
# ---------------------------------------------------------------------------

def _train_member(
    member:       EnsembleMember,
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    X_val:        np.ndarray,
    y_val:        np.ndarray,
    member_idx:   int,
    epochs:       int   = MAX_EPOCHS,
    batch_size:   int   = BATCH_SIZE,
    lr:           float = LR,
    weight_decay: float = 1e-4,
    patience:     int   = PATIENCE,
) -> Dict[str, list]:
    member = member.to(DEVICE)

    def _loader(X, y, shuffle):
        return DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            ),
            batch_size=batch_size, shuffle=shuffle, num_workers=0,
        )

    train_loader = _loader(X_train, y_train, True)
    val_loader   = _loader(X_val,   y_val,   False)

    optimizer = torch.optim.Adam(member.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10)

    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        member.train()
        tr = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            m, lv = member(X_b)
            loss = _heteroscedastic_nll(m, lv, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(member.parameters(), 1.0)
            optimizer.step()
            tr += loss.item() * len(X_b)
        tr /= len(train_loader.dataset)

        member.eval()
        vl = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                m, lv = member(X_b)
                vl += _heteroscedastic_nll(m, lv, y_b).item() * len(X_b)
        vl /= len(val_loader.dataset)

        scheduler.step(vl)
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in member.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [ensemble member {member_idx}] Early stop epoch {epoch} | best_val={best_val:.4f}")
                break

    if best_state:
        member.load_state_dict(best_state)
    return history


# ---------------------------------------------------------------------------
# 3. Train the Full Ensemble
# ---------------------------------------------------------------------------

def train_ensemble(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    n_members:  int = N_ENSEMBLE,
    input_dim:  int = 800,
    n_targets:  int = 3,
    **train_kwargs,
) -> Tuple[List[EnsembleMember], List[Dict]]:
    """
    Train N independent ensemble members and return them with their histories.

    Each member is initialised with a different random seed to ensure diversity.
    """
    members   = []
    histories = []

    for i in range(n_members):
        print(f"\n[ensemble] Training member {i+1}/{n_members} ...")
        # Different seed per member for diverse initialisations
        torch.manual_seed(RANDOM_SEED + i)
        np.random.seed(RANDOM_SEED + i)

        member = EnsembleMember(input_dim=input_dim, n_targets=n_targets)
        hist   = _train_member(member, X_train, y_train, X_val, y_val,
                               member_idx=i + 1, **train_kwargs)
        members.append(member)
        histories.append(hist)
        print(f"  [ensemble member {i+1}] Done | best_val={min(hist['val_loss']):.4f}")

    return members, histories


# ---------------------------------------------------------------------------
# 4. Ensemble Inference
# ---------------------------------------------------------------------------

def ensemble_predict(
    members:    List[EnsembleMember],
    X_test:     np.ndarray,
    batch_size: int = 128,
) -> Dict[str, np.ndarray]:
    """
    Aggregate predictions from all ensemble members.

    Mixture-of-Gaussians combination:
        μ* = mean of member means
        σ*² = mean(σ_i² + μ_i²) - μ*²   (law of total variance)

    Returns per-target:
        mean_<name>      : Ensemble predictive mean
        epistemic_<name> : Std of member means (spread of the ensemble)
        aleatoric_<name> : Mean of member σ (average data uncertainty)
        total_std_<name> : Sqrt(total variance)
    """
    all_means    = []
    all_log_vars = []

    for member in members:
        member.eval().to(DEVICE)
        loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False,
        )
        m_list, lv_list = [], []
        with torch.no_grad():
            for (X_b,) in loader:
                m, lv = member(X_b.to(DEVICE))
                m_list.append(m.cpu().numpy())
                lv_list.append(lv.cpu().numpy())
        all_means.append(np.concatenate(m_list,  axis=0))    # (N, n_targets)
        all_log_vars.append(np.concatenate(lv_list, axis=0))

    all_means    = np.stack(all_means,    axis=0)   # (M, N, n_targets)
    all_log_vars = np.stack(all_log_vars, axis=0)   # (M, N, n_targets)

    n_targets    = all_means.shape[2]
    target_names = TARGET_COLS[:n_targets]

    results = {"all_means": all_means, "all_log_vars": all_log_vars}

    for i, name in enumerate(target_names):
        means_i    = all_means[:, :, i]          # (M, N)
        log_vars_i = all_log_vars[:, :, i]       # (M, N)
        vars_i     = np.exp(log_vars_i)          # (M, N) — per-member variance

        mu_star      = means_i.mean(axis=0)                                  # (N,)
        total_var    = (vars_i + means_i**2).mean(axis=0) - mu_star**2       # (N,)
        total_var    = np.maximum(total_var, 0.0)                            # numerical safety
        epistemic    = means_i.std(axis=0)                                   # (N,)
        aleatoric    = np.sqrt(vars_i).mean(axis=0)                          # (N,)
        total_std    = np.sqrt(total_var)                                     # (N,)

        results[f"mean_{name}"]      = mu_star
        results[f"epistemic_{name}"] = epistemic
        results[f"aleatoric_{name}"] = aleatoric
        results[f"total_std_{name}"] = total_std

    return results


# ---------------------------------------------------------------------------
# 5. Save Outputs
# ---------------------------------------------------------------------------

def save_members(members: List[EnsembleMember], run_tag: str = "ensemble") -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, m in enumerate(members):
        path = MODELS_DIR / f"{run_tag}_member{i+1}_{ts}.pt"
        torch.save(m.state_dict(), path)
    print(f"[ensemble] {len(members)} members saved to {MODELS_DIR}")


def load_members(
    paths: List[Path],
    input_dim:   int = 800,
    hidden_dims: tuple = (256, 128, 64),
    n_targets:   int = 3,
) -> List[EnsembleMember]:
    """Load saved ensemble members from disk."""
    members = []
    for p in sorted(paths):
        m = EnsembleMember(input_dim=input_dim, hidden_dims=hidden_dims, n_targets=n_targets)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m = m.to(DEVICE)
        m.eval()
        members.append(m)
    print(f"[ensemble] Loaded {len(members)} members from {paths[0].parent}")
    return members


def save_learning_curves(
    histories: List[Dict], run_tag: str = "ensemble"
) -> Path:
    """
    Plot validation NLL loss curves for all ensemble members.

    Each member is initialised with a different random seed, producing
    diverse weight solutions.  Curves that converge to similar loss values
    but diverge in trajectory confirm that the ensemble captures genuine
    functional diversity — a prerequisite for reliable epistemic uncertainty
    estimates.
    """
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    n      = len(histories)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (hist, col) in enumerate(zip(histories, colors)):
        epochs = range(1, len(hist["val_loss"]) + 1)
        ax.plot(epochs, hist["val_loss"], color=col,
                linewidth=2.2, alpha=0.88, label=f"Member {i+1}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation NLL Loss")
    ax.set_title("Deep Ensemble — Per-Member Validation Loss")
    ax.legend(ncol=2, fontsize=12)

    path = FIGURES_DIR / f"{run_tag}_learning_curves_{ts}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[ensemble] Learning curves → {path}")
    return path


def save_uncertainty_plots(
    results:      Dict,
    y_test:       np.ndarray,
    target_names: List[str],
    run_tag:      str = "ensemble",
) -> None:
    """
    Three-panel uncertainty analysis figure for each regression target.
    Identical layout to MC Dropout to allow direct side-by-side comparison.

    Panel 1 — Predicted vs Ground Truth (colour = σ_total):
      Warm colours indicate predictions the ensemble is unsure about.
      Systematic scatter around the diagonal quantifies generalisation error.

    Panel 2 — Epistemic vs Aleatoric Uncertainty:
      For Deep Ensembles, epistemic uncertainty is captured by disagreement
      among the 5 members (variance of their means).  Aleatoric uncertainty
      is the mean of each member's predicted output variance.

    Panel 3 — Distribution of Total σ:
      Shows the spread of ensemble confidence across the test set.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, name in enumerate(target_names):
        y_true    = y_test[:, i]
        y_mean    = results[f"mean_{name}"]
        epistemic = results[f"epistemic_{name}"]
        aleatoric = results[f"aleatoric_{name}"]
        total_std = results[f"total_std_{name}"]
        unit_s    = TARGET_UNITS.get(name, "")
        disp      = TARGET_DISPLAY.get(name, name)
        lbl       = f"({unit_s})" if unit_s else ""

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Deep Ensemble — Uncertainty Analysis: {disp}")

        # ── Panel 1: Predicted vs True ──────────────────────────────────────
        ax = axes[0]
        sc = ax.scatter(
            y_true, y_mean, c=total_std,
            cmap="YlOrRd", alpha=0.85, s=50,
            edgecolors="white", linewidths=0.4,
            vmin=total_std.min(), vmax=np.percentile(total_std, 95),
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"Total σ {lbl}", fontsize=13)
        cbar.ax.tick_params(labelsize=12)

        lim = [
            min(y_true.min(), y_mean.min()) * 0.96,
            max(y_true.max(), y_mean.max()) * 1.04,
        ]
        ax.plot(lim, lim, "--", color=PALETTE["perfect"],
                linewidth=2.0, label="Perfect prediction")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"Ground Truth — {disp} {lbl}")
        ax.set_ylabel(f"Predicted — {disp} {lbl}")
        ax.set_title("Prediction vs. Ground Truth\n(colour = total uncertainty)")
        rmse = float(np.sqrt(np.mean((y_true - y_mean) ** 2)))
        ax.text(0.04, 0.97, f"RMSE = {rmse:.3f} {unit_s}",
                transform=ax.transAxes, va="top", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))
        ax.legend(fontsize=12)

        # ── Panel 2: Epistemic vs Aleatoric ─────────────────────────────────
        ax = axes[1]
        ax.scatter(epistemic, aleatoric, alpha=0.65, s=45,
                   color=PALETTE["ensemble"], edgecolors="white", linewidths=0.4)
        ax.set_xlabel(f"Epistemic σ — member disagreement {lbl}")
        ax.set_ylabel(f"Aleatoric σ — data/sensor noise {lbl}")
        ax.set_title("Epistemic vs. Aleatoric Uncertainty\nDecomposition")

        frac_aleatoric = float(np.mean(aleatoric) / (np.mean(epistemic) + np.mean(aleatoric) + 1e-9))
        ax.text(0.97, 0.97,
                f"Aleatoric share:\n{frac_aleatoric*100:.1f}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))

        # ── Panel 3: Total σ histogram ───────────────────────────────────────
        ax = axes[2]
        ax.hist(total_std, bins=28, color=PALETTE["ensemble"],
                edgecolor="white", alpha=0.85, linewidth=0.6)
        ax.axvline(total_std.mean(), color=PALETTE["high_risk"],
                   linestyle="--", linewidth=2.5,
                   label=f"Mean σ = {total_std.mean():.3f} {unit_s}")
        ax.axvline(np.median(total_std), color=PALETTE["rf"],
                   linestyle=":", linewidth=2.2,
                   label=f"Median σ = {np.median(total_std):.3f} {unit_s}")
        ax.set_xlabel(f"Total Predictive σ {lbl}")
        ax.set_ylabel("Number of Test Samples")
        ax.set_title("Distribution of Total\nPredictive Uncertainty")
        ax.legend()

        path = FIGURES_DIR / f"{run_tag}_uncertainty_{name}_{ts}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"[ensemble] Uncertainty plot → {path}")


def save_metrics(
    results:      Dict,
    y_test:       np.ndarray,
    target_names: List[str],
    run_tag:      str = "ensemble",
    extra_info:   str = "",
) -> Path:
    from evaluation.metrics import full_uncertainty_report

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = METRICS_DIR / f"{run_tag}_metrics_{ts}.txt"

    lines = [
        "=" * 65,
        f"Deep Ensemble ({N_ENSEMBLE} members) — Uncertainty-Aware Regression",
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        extra_info,
        "=" * 65, "",
    ]

    for i, name in enumerate(target_names):
        y_true    = y_test[:, i]
        y_mean    = results[f"mean_{name}"]
        total_std = results[f"total_std_{name}"]
        epistemic = results[f"epistemic_{name}"]
        aleatoric = results[f"aleatoric_{name}"]
        report    = full_uncertainty_report(y_true, y_mean, total_std, target_name=name)

        lines += [
            f"Target: {name}",
            f"  RMSE      : {report['RMSE']:.4f}",
            f"  MAE       : {report['MAE']:.4f}",
            f"  R²        : {report['R2']:.4f}",
            f"  PICP 95%  : {report['PICP_95']:.4f}  (ideal = 0.95)",
            f"  MPIW 95%  : {report['MPIW_95']:.4f}",
            f"  ECE       : {report['ECE']:.4f}  (ideal = 0.0)",
            f"  NLL       : {report['NLL']:.4f}",
            f"  Mean epistemic σ : {epistemic.mean():.4f}",
            f"  Mean aleatoric σ : {aleatoric.mean():.4f}",
            "",
        ]

    path.write_text("\n".join(lines))
    print(f"[ensemble] Metrics saved → {path}")
    return path
