"""
mc_dropout.py
=============
Monte Carlo Dropout Neural Network for Uncertainty-Aware Regression.

Thesis: Uncertainty-Aware Machine Learning for Microwave-Based Body Composition Assessment
Task:   Multi-output regression — predict [Skin_mm, Fat_mm, Muscle_cm2] from
        800-dimensional S-parameter feature vectors.

Theory (Gal & Ghahramani, 2016):
  Keeping Dropout active at inference time is equivalent to approximate Bayesian
  inference over the network weights. Each stochastic forward pass samples a
  different thinned sub-network from the approximate posterior, and the empirical
  statistics across T passes estimate the posterior predictive distribution.

Uncertainty decomposition (Kendall & Gal, 2017):
  - Epistemic (model) uncertainty: variance of predicted means across T passes.
    Reflects lack of knowledge — reducible with more data.
  - Aleatoric (data) uncertainty: mean of predicted variances across T passes.
    Reflects irreducible measurement noise in the S-parameter signals.
  - Total predictive uncertainty: epistemic + aleatoric.

Architecture:
  Shared encoder: Linear(800→256) → BN → ReLU → Dropout
                  Linear(256→128) → BN → ReLU → Dropout
                  Linear(128→64)  → BN → ReLU → Dropout
  Per-target heads (×3): Linear(64→2) → [mean, log_variance]

Loss (heteroscedastic Gaussian NLL per target, Kendall & Gal 2017):
  L = Σ_k [ (y_k - μ_k)² / (2·exp(log_var_k)) + log_var_k / 2 ]

References:
  Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. ICML.
  Kendall & Gal (2017). What Uncertainties Do We Need in Bayesian Deep Learning? NeurIPS.
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

# Apply publication-quality style
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.plot_style import apply_pub_style, PALETTE, TARGET_UNITS, TARGET_DISPLAY
apply_pub_style()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RANDOM_SEED, BATCH_SIZE, MAX_EPOCHS, LR, PATIENCE,
    DROPOUT_P, MC_T, MODELS_DIR, FIGURES_DIR, METRICS_DIR, TARGET_COLS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Model Architecture
# ---------------------------------------------------------------------------

class MCDropoutRegressor(nn.Module):
    """
    Shared-encoder, multi-head MC Dropout network for regression with
    heteroscedastic uncertainty.

    For each of the `n_targets` output targets the network predicts:
        - μ   (mean prediction)
        - log σ² (log variance — aleatoric uncertainty)

    The Dropout layers are kept ACTIVE during inference by calling
    mc_forward() instead of model.eval() + forward(). BatchNorm layers
    are set to eval mode during inference to use running statistics.

    Args:
        input_dim  : Number of input features (default 800).
        hidden_dims: Encoder hidden layer widths.
        n_targets  : Number of regression targets (default 3).
        dropout_p  : Dropout probability (applied after every hidden block).
    """

    def __init__(
        self,
        input_dim:   int = 800,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        n_targets:   int = 3,
        dropout_p:   float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.n_targets = n_targets
        self.dropout_p = dropout_p

        # --- Shared encoder ---
        encoder_layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            encoder_layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ]
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Per-target heads: predict (mean, log_var) ---
        self.heads = nn.ModuleList([
            nn.Linear(in_dim, 2) for _ in range(n_targets)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass (Dropout active — same behaviour as mc_forward
        since we never call model.eval() explicitly except in validation).

        Returns:
            means    : (N, n_targets) — predicted means
            log_vars : (N, n_targets) — predicted log variances
        """
        h = self.encoder(x)
        outputs = [head(h) for head in self.heads]   # list of (N, 2)
        stacked = torch.stack(outputs, dim=1)         # (N, n_targets, 2)
        means    = stacked[:, :, 0]                   # (N, n_targets)
        log_vars = stacked[:, :, 1]                   # (N, n_targets)
        return means, log_vars

    def mc_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stochastic forward pass: BatchNorm in eval mode, Dropout in train mode.
        Call this T times at inference to sample from the posterior.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
            elif isinstance(module, nn.Dropout):
                module.train()
        return self.forward(x)


# ---------------------------------------------------------------------------
# 2. Heteroscedastic Loss
# ---------------------------------------------------------------------------

def heteroscedastic_nll_loss(
    means:    torch.Tensor,   # (N, n_targets)
    log_vars: torch.Tensor,   # (N, n_targets)
    targets:  torch.Tensor,   # (N, n_targets)
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss with learned variance.

    L = mean over samples and targets of:
        [ (y - μ)² / (2·exp(log_σ²)) + log_σ² / 2 ]

    The network learns to increase log_σ² for hard-to-predict samples,
    trading off prediction accuracy for uncertainty calibration.
    Clamping log_var prevents numerical instability.
    """
    log_vars = torch.clamp(log_vars, min=-10.0, max=10.0)
    precision = torch.exp(-log_vars)                       # 1/σ²
    loss = precision * (targets - means) ** 2 + log_vars  # per element
    return loss.mean()


# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------

def train(
    model:        MCDropoutRegressor,
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    X_val:        np.ndarray,
    y_val:        np.ndarray,
    epochs:       int   = MAX_EPOCHS,
    batch_size:   int   = BATCH_SIZE,
    lr:           float = LR,
    weight_decay: float = 1e-4,
    patience:     int   = PATIENCE,
    verbose:      bool  = True,
) -> Dict[str, list]:
    """
    Train MCDropoutRegressor with heteroscedastic NLL loss.

    Early stopping monitors validation loss. Best weights are restored
    at the end of training. LR is halved when validation loss plateaus.

    Returns:
        history: dict with 'train_loss' and 'val_loss' lists.
    """
    model = model.to(DEVICE)

    def _make_loader(X, y, shuffle):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        return DataLoader(
            TensorDataset(Xt, yt),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
        )

    train_loader = _make_loader(X_train, y_train, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss   = float("inf")
    best_state      = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            means, log_vars = model(X_b)
            loss = heteroscedastic_nll_loss(means, log_vars, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(train_loader.dataset)

        # Validation (deterministic: full eval mode)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                means, log_vars = model(X_b)
                val_loss += heteroscedastic_nll_loss(means, log_vars, y_b).item() * len(X_b)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>4}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} | best val_loss={best_val_loss:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ---------------------------------------------------------------------------
# 4. MC Dropout Inference
# ---------------------------------------------------------------------------

def mc_predict(
    model:      MCDropoutRegressor,
    X_test:     np.ndarray,
    T:          int = MC_T,
    batch_size: int = 128,
) -> Dict[str, np.ndarray]:
    """
    Run T stochastic forward passes and compute uncertainty estimates.

    Returns a dict with per-target arrays (shape: (N_test,) each):
        mean_<target>      : Predictive mean (final point estimate)
        epistemic_<target> : Epistemic (model) uncertainty — std of means across T passes
        aleatoric_<target> : Aleatoric (data) uncertainty — mean of predicted σ across T passes
        total_std_<target> : Total uncertainty = sqrt(epistemic² + aleatoric²)
        all_means          : np.ndarray (T, N, n_targets) — all stochastic means
        all_log_vars       : np.ndarray (T, N, n_targets) — all stochastic log vars
    """
    model = model.to(DEVICE)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    loader   = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    all_means    = []   # will be (T, N, n_targets)
    all_log_vars = []

    for _ in range(T):
        pass_means, pass_lvars = [], []
        with torch.no_grad():
            for (X_b,) in loader:
                X_b = X_b.to(DEVICE)
                m, lv = model.mc_forward(X_b)
                pass_means.append(m.cpu().numpy())
                pass_lvars.append(lv.cpu().numpy())
        all_means.append(np.concatenate(pass_means, axis=0))
        all_log_vars.append(np.concatenate(pass_lvars, axis=0))

    all_means    = np.stack(all_means,    axis=0)   # (T, N, n_targets)
    all_log_vars = np.stack(all_log_vars, axis=0)   # (T, N, n_targets)

    results = {
        "all_means":    all_means,
        "all_log_vars": all_log_vars,
    }

    n_targets = all_means.shape[2]
    target_names = TARGET_COLS[:n_targets]

    for i, name in enumerate(target_names):
        means_i    = all_means[:, :, i]          # (T, N)
        log_vars_i = all_log_vars[:, :, i]       # (T, N)

        pred_mean   = means_i.mean(axis=0)                    # (N,)
        epistemic   = means_i.std(axis=0)                     # (N,) — std of means
        aleatoric   = np.exp(log_vars_i * 0.5).mean(axis=0)  # (N,) — mean of σ
        total_std   = np.sqrt(epistemic**2 + aleatoric**2)   # (N,)

        results[f"mean_{name}"]      = pred_mean
        results[f"epistemic_{name}"] = epistemic
        results[f"aleatoric_{name}"] = aleatoric
        results[f"total_std_{name}"] = total_std

    return results


# ---------------------------------------------------------------------------
# 5. Save Outputs
# ---------------------------------------------------------------------------

def save_model(model: MCDropoutRegressor, run_tag: str = "mc_dropout") -> Path:
    """Save model state_dict to outputs/models/."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODELS_DIR / f"{run_tag}_{ts}.pt"
    torch.save(model.state_dict(), path)
    print(f"[mc_dropout] Model saved → {path}")
    return path


def load_model(path: Path, **kwargs) -> MCDropoutRegressor:
    model = MCDropoutRegressor(**kwargs)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE)


def save_learning_curves(history: Dict, run_tag: str = "mc_dropout") -> Path:
    """
    Plot training and validation NLL loss over epochs.

    The heteroscedastic Gaussian NLL (negative log-likelihood) is the
    training objective: the model simultaneously learns to predict tissue
    thickness (mean) and its own uncertainty (variance).  Convergence of
    both curves without divergence confirms stable training.
    """
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(9, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=PALETTE["mc"],
            linewidth=2.5, label="Training NLL")
    ax.plot(epochs, history["val_loss"],   color=PALETTE["high_risk"],
            linewidth=2.5, linestyle="--", label="Validation NLL")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Heteroscedastic Gaussian NLL")
    ax.set_title("MC Dropout — Training and Validation Loss")
    ax.legend()

    # Mark best validation epoch
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val   = min(history["val_loss"])
    ax.axvline(best_epoch, color=PALETTE["threshold"], linestyle=":",
               linewidth=1.8, alpha=0.8, label=f"Best epoch ({best_epoch})")
    ax.scatter([best_epoch], [best_val], s=80, zorder=5,
               color=PALETTE["high_risk"], edgecolors="white", linewidths=1.2)
    ax.legend()

    path = FIGURES_DIR / f"{run_tag}_learning_curves_{ts}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[mc_dropout] Learning curves → {path}")
    return path


def save_uncertainty_plots(
    results:      Dict,
    y_test:       np.ndarray,
    target_names: List[str],
    run_tag:      str = "mc_dropout",
) -> None:
    """
    Three-panel uncertainty analysis figure for each regression target.

    Panel 1 — Predicted vs Ground Truth (colour = σ_total):
      Each point is one test sample.  Warm colours (high σ) highlight
      predictions the model is unsure about; cool colours mark confident
      predictions.  Systematic deviation from the diagonal quantifies bias.

    Panel 2 — Epistemic vs Aleatoric Uncertainty:
      Epistemic uncertainty (variance of T=50 mean predictions) reflects
      lack of training data and is theoretically reducible.  Aleatoric
      uncertainty (mean of T=50 predicted variances) reflects irreducible
      S-parameter noise.  Fat typically dominates in aleatoric; muscle
      has higher epistemic spread due to anatomical variability.

    Panel 3 — Distribution of Total Predictive σ:
      Shows how confidence is distributed across all test samples.
      The vertical red line marks the mean σ; the shape indicates
      whether uncertainty is uniform or concentrated in hard cases.
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
        fig.suptitle(f"MC Dropout — Uncertainty Analysis: {disp}")

        # ── Panel 1: Predicted vs True coloured by σ_total ─────────────────
        ax = axes[0]
        scatter = ax.scatter(
            y_true, y_mean, c=total_std,
            cmap="YlOrRd", alpha=0.85, s=50,
            edgecolors="white", linewidths=0.4,
            vmin=total_std.min(), vmax=np.percentile(total_std, 95),
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
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

        # ── Panel 2: Epistemic vs Aleatoric ────────────────────────────────
        ax = axes[1]
        ax.scatter(epistemic, aleatoric, alpha=0.65, s=45,
                   color=PALETTE["mc"], edgecolors="white", linewidths=0.4)
        ax.set_xlabel(f"Epistemic σ — model uncertainty {lbl}")
        ax.set_ylabel(f"Aleatoric σ — data/sensor noise {lbl}")
        ax.set_title("Epistemic vs. Aleatoric Uncertainty\nDecomposition")

        # Annotate dominant component
        frac_aleatoric = float(np.mean(aleatoric) / (np.mean(epistemic) + np.mean(aleatoric) + 1e-9))
        ax.text(0.97, 0.97,
                f"Aleatoric share:\n{frac_aleatoric*100:.1f}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))

        # ── Panel 3: Total σ histogram ─────────────────────────────────────
        ax = axes[2]
        ax.hist(total_std, bins=28, color=PALETTE["mc"],
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
        print(f"[mc_dropout] Uncertainty plot → {path}")


def save_metrics(
    results:      Dict,
    y_test:       np.ndarray,
    target_names: List[str],
    run_tag:      str = "mc_dropout",
    extra_info:   str = "",
) -> Path:
    """Compute and save all uncertainty metrics to outputs/metrics/."""
    from evaluation.metrics import full_uncertainty_report

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = METRICS_DIR / f"{run_tag}_metrics_{ts}.txt"

    lines = [
        "=" * 65,
        "MC Dropout — Uncertainty-Aware Regression Results",
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"T (stochastic passes) = {MC_T}",
        extra_info,
        "=" * 65,
        "",
    ]

    import pandas as pd
    rows = []
    for i, name in enumerate(target_names):
        y_true    = y_test[:, i]
        y_mean    = results[f"mean_{name}"]
        total_std = results[f"total_std_{name}"]
        report    = full_uncertainty_report(y_true, y_mean, total_std, target_name=name)
        rows.append(report)

        epistemic = results[f"epistemic_{name}"]
        aleatoric = results[f"aleatoric_{name}"]
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
    print(f"[mc_dropout] Metrics saved → {path}")
    return path
