"""
uncertainty.py
==============
Monte Carlo (MC) Dropout Neural Network for Uncertainty-Aware Binary Classification.

Thesis: Uncertainty-Aware Machine Learning for Microwave-Based Body Composition Assessment
Task:   Binary classification (Tumor vs. Reference) from microwave S-parameter features.
        Input dimension: 604 (S11 + S21 frequency sweep features)

Key References:
  - Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation: Representing Model
    Uncertainty in Deep Learning." ICML 2016.
  - Kendall & Gal (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for
    Computer Vision?" NeurIPS 2017.

Design:
  - MCDropoutClassifier: nn.Module with Dropout layers kept ACTIVE at inference via a
    dedicated mc_dropout_forward() method. This enables stochastic forward passes that
    approximate sampling from the posterior predictive distribution.
  - train_model(): full training loop with BCE loss, Adam optimizer, DataLoaders,
    early stopping, and LR scheduling.
  - mc_predict(): T stochastic passes → mean probability (prediction) + variance
    (epistemic uncertainty proxy).
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — safe for scripts with no display
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Dict

# Resolve OUTPUT_DIR from config.py regardless of how the module is imported
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
from config import OUTPUT_DIR


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Model Architecture
# ---------------------------------------------------------------------------

class MCDropoutClassifier(nn.Module):
    """
    Fully Connected Neural Network with Dropout for MC Dropout inference.

    Architecture (default):
        Linear(604 → 512) → BatchNorm → ReLU → Dropout(p)
        Linear(512 → 256) → BatchNorm → ReLU → Dropout(p)
        Linear(256 → 128) → BatchNorm → ReLU → Dropout(p)
        Linear(128 →  64) → BatchNorm → ReLU → Dropout(p)
        Linear( 64 →   1) → Sigmoid

    The key design principle: Dropout layers are NOT wrapped with model.eval() during
    inference. Instead, mc_dropout_forward() explicitly sets only BatchNorm layers to
    eval mode while leaving Dropout stochastic, enabling Bayesian approximation.

    Args:
        input_dim  : Number of input features (default 604).
        hidden_dims: Sequence of hidden layer widths.
        dropout_p  : Dropout probability applied after every hidden block.
    """

    def __init__(
        self,
        input_dim: int = 604,
        hidden_dims: Tuple[int, ...] = (512, 256, 128, 64),
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout_p = dropout_p
        layers = []
        in_dim = input_dim

        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ]
            in_dim = out_dim

        # Final projection to a single logit for binary classification
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (used during training)."""
        return torch.sigmoid(self.network(x))

    def mc_dropout_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stochastic forward pass for MC Dropout inference.

        Sets BatchNorm layers to eval mode (to use running statistics, not batch stats)
        while deliberately leaving Dropout layers in training mode so they continue
        to randomly zero activations. This is the core of the Gal & Ghahramani (2016)
        approximation to Bayesian inference.

        NOTE: Do NOT call model.eval() before this method — that would disable Dropout
        and collapse all T passes into identical deterministic outputs.
        """
        # Freeze BatchNorm statistics; keep Dropout stochastic
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
            elif isinstance(module, nn.Dropout):
                module.train()

        return torch.sigmoid(self.network(x))


# ---------------------------------------------------------------------------
# 2. Dataset Helper
# ---------------------------------------------------------------------------

def _build_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """Converts NumPy arrays to PyTorch TensorDatasets and wraps them in DataLoaders."""

    def _to_tensor_dataset(X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        # BCELoss expects float targets of shape (N, 1)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return TensorDataset(X_t, y_t)

    train_loader = DataLoader(
        _to_tensor_dataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        _to_tensor_dataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------

def train_model(
    model: MCDropoutClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train the MCDropoutClassifier.

    Strategy:
      - Binary Cross-Entropy loss (BCELoss) for single-output sigmoid networks.
      - Adam optimizer with L2 weight decay as an additional regularizer.
      - ReduceLROnPlateau halves the LR when validation loss stalls.
      - Early stopping restores the best checkpoint and halts training when
        val loss has not improved for `patience` consecutive epochs, preventing
        overfitting on the small microwave dataset.

    Args:
        model       : MCDropoutClassifier instance (already moved to DEVICE).
        X_train     : Scaled training features, shape (N_train, 604).
        y_train     : Binary labels {0, 1}, shape (N_train,).
        X_val       : Scaled validation features, shape (N_val, 604).
        y_val       : Binary labels {0, 1}, shape (N_val,).
        epochs      : Maximum number of training epochs.
        batch_size  : Mini-batch size for SGD.
        lr          : Initial Adam learning rate.
        weight_decay: L2 regularisation coefficient for Adam.
        patience    : Early-stopping patience (epochs without val improvement).
        verbose     : Print per-epoch metrics when True.

    Returns:
        history: Dict with keys 'train_loss', 'val_loss', 'val_acc' — lists of
                 per-epoch values for plotting learning curves.
    """
    model = model.to(DEVICE)
    train_loader, val_loader = _build_loaders(X_train, y_train, X_val, y_val, batch_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------------
        # Training phase
        # ------------------------------------------------------------------
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)          # Standard forward pass with Dropout
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ------------------------------------------------------------------
        # Validation phase — deterministic (model.eval() disables Dropout)
        # ------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * X_batch.size(0)
                all_preds.append((preds.cpu() >= 0.5).float())
                all_targets.append(y_batch.cpu())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(
            torch.cat(all_targets).numpy(),
            torch.cat(all_preds).numpy(),
        )

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:>4}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}%"
            )

        # ------------------------------------------------------------------
        # Early stopping — keep the best checkpoint in memory
        # ------------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Deep-copy weights to CPU to avoid GPU memory accumulation
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}. Restoring best weights.")
                break

    # Restore the best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if verbose:
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    return history


# ---------------------------------------------------------------------------
# 4. MC Dropout Inference
# ---------------------------------------------------------------------------

def mc_predict(
    model: MCDropoutClassifier,
    X_test: np.ndarray,
    T: int = 50,
    batch_size: int = 128,
) -> Dict[str, np.ndarray]:
    """
    Perform T stochastic forward passes to estimate predictive mean and variance.

    Under the MC Dropout approximation (Gal & Ghahramani, 2016), each pass samples
    a different thinned sub-network by randomly dropping units. The empirical
    statistics across T passes approximate the posterior predictive distribution:

        p(y=1 | x, X_train) ≈ (1/T) Σ_t  sigmoid(f^t(x))

    where f^t denotes the t-th stochastic forward pass.

    Uncertainty decomposition:
      - Mean  (μ): The expected prediction — use as the final class probability.
      - Variance (σ²): Proxy for *epistemic uncertainty* — high variance signals
        that the model has insufficient knowledge about a sample (e.g., out-of-
        distribution, ambiguous, or adversarial inputs). Ideally flagged for
        clinician review in a medical context.

    Args:
        model     : Trained MCDropoutClassifier.
        X_test    : Scaled test features, shape (N_test, 604).
        T         : Number of stochastic forward passes (default 50). Higher T
                    reduces Monte Carlo estimation error at the cost of compute.
        batch_size: Mini-batch size for inference (avoids OOM on large datasets).

    Returns:
        A dict with:
          'mean_prob'  : np.ndarray shape (N_test,) — mean predictive probability
                         (use >= 0.5 threshold for class prediction).
          'variance'   : np.ndarray shape (N_test,) — predictive variance per sample
                         (epistemic uncertainty score).
          'std'        : np.ndarray shape (N_test,) — standard deviation (√variance),
                         more interpretable for reporting confidence intervals.
          'all_passes' : np.ndarray shape (T, N_test) — raw per-pass probabilities
                         for downstream calibration or visualisation.
    """
    model = model.to(DEVICE)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Accumulate predictions across all T passes: shape (T, N_test)
    all_pass_probs = []

    for _ in range(T):
        pass_probs = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(DEVICE)
                # mc_dropout_forward keeps Dropout ON and BatchNorm in eval mode
                probs = model.mc_dropout_forward(X_batch)   # (batch, 1)
                pass_probs.append(probs.squeeze(1).cpu().numpy())

        all_pass_probs.append(np.concatenate(pass_probs))   # (N_test,)

    all_pass_probs = np.stack(all_pass_probs, axis=0)        # (T, N_test)

    mean_prob = all_pass_probs.mean(axis=0)                  # (N_test,)
    variance  = all_pass_probs.var(axis=0)                   # (N_test,)
    std       = np.sqrt(variance)                            # (N_test,)

    return {
        "mean_prob":   mean_prob,
        "variance":    variance,
        "std":         std,
        "all_passes":  all_pass_probs,
    }


# ---------------------------------------------------------------------------
# 5. Evaluation Helper
# ---------------------------------------------------------------------------

def evaluate_with_uncertainty(
    results: Dict[str, np.ndarray],
    y_true: np.ndarray,
    uncertainty_threshold: float = 0.05,
    target_names: Tuple[str, ...] = ("Reference", "Tumor"),
) -> None:
    """
    Print a classification report and flag high-uncertainty predictions.

    Samples whose predictive variance exceeds `uncertainty_threshold` are
    considered "uncertain" and would, in a clinical deployment, be routed to
    a human expert rather than acted upon automatically.

    Args:
        results              : Output dict from mc_predict().
        y_true               : Ground-truth labels, shape (N_test,).
        uncertainty_threshold: Variance threshold above which a prediction is
                               flagged as uncertain (tune via calibration).
        target_names         : Class names for the classification report.
    """
    mean_prob = results["mean_prob"]
    variance  = results["variance"]

    y_pred = (mean_prob >= 0.5).astype(int)
    uncertain_mask = variance > uncertainty_threshold

    print("=" * 60)
    print("MC Dropout Inference Results")
    print("=" * 60)
    print(f"Total samples      : {len(y_true)}")
    print(f"Uncertain samples  : {uncertain_mask.sum()} "
          f"({uncertain_mask.mean() * 100:.1f}%, variance > {uncertainty_threshold})")
    print(f"\n--- All Predictions ---")
    print(classification_report(y_true, y_pred, target_names=list(target_names)))

    # Report accuracy on the CERTAIN subset (model's confident predictions)
    certain_mask = ~uncertain_mask
    if certain_mask.sum() > 0:
        certain_acc = accuracy_score(y_true[certain_mask], y_pred[certain_mask])
        print(f"--- Certain Predictions ({certain_mask.sum()} samples) ---")
        print(f"Accuracy (certain only): {certain_acc * 100:.2f}%")
        print(classification_report(
            y_true[certain_mask], y_pred[certain_mask],
            target_names=list(target_names)
        ))

    # Summary statistics on uncertainty scores
    print("--- Uncertainty Score Statistics ---")
    print(f"  Mean variance : {variance.mean():.5f}")
    print(f"  Max variance  : {variance.max():.5f}")
    print(f"  Min variance  : {variance.min():.5f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. Output Persistence  (metrics log + plots → OUTPUT_DIR)
# ---------------------------------------------------------------------------

def save_outputs(
    results: Dict[str, np.ndarray],
    y_true: np.ndarray,
    history: Dict[str, list],
    uncertainty_threshold: float = 0.05,
    target_names: Tuple[str, ...] = ("Reference", "Tumor"),
    run_tag: str = "mc_dropout",
) -> None:
    """
    Persist all run artefacts to OUTPUT_DIR with a shared timestamp prefix.

    Files written
    -------------
    <run_tag>_metrics_<timestamp>.txt
        Plain-text classification report, uncertainty statistics, and training
        hyper-parameter summary — mirrors the format used by baseline runs.

    <run_tag>_learning_curves_<timestamp>.png
        Two-panel figure: (left) train vs. val BCE loss, (right) val accuracy.
        Useful for diagnosing overfitting and early-stopping behaviour.

    <run_tag>_confusion_matrix_<timestamp>.png
        Seaborn heatmap of the confusion matrix on the test set (certain +
        uncertain predictions combined), matching the style of baseline heatmaps.

    <run_tag>_uncertainty_<timestamp>.png
        Three-panel figure showing the epistemic uncertainty distribution:
          · Histogram of per-sample predictive variance coloured by true class.
          · Scatter of mean probability vs. variance (reveals the high-conf /
            high-uncertainty quadrant where false positives hide).
          · Bar chart: accuracy on certain vs. uncertain subsets.

    <run_tag>_model_<timestamp>.pt
        PyTorch state_dict checkpoint for the trained model.

    Args:
        results              : Output dict from mc_predict().
        y_true               : Integer ground-truth labels, shape (N_test,).
        history              : Dict returned by train_model() with keys
                               'train_loss', 'val_loss', 'val_acc'.
        uncertainty_threshold: Variance threshold separating certain/uncertain.
        target_names         : Ordered class labels matching label encoding.
        run_tag              : Prefix for all output filenames (default 'mc_dropout').
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix    = OUTPUT_DIR / f"{run_tag}_{timestamp}"

    mean_prob      = results["mean_prob"]
    variance       = results["variance"]
    y_pred         = (mean_prob >= 0.5).astype(int)
    uncertain_mask = variance > uncertainty_threshold
    certain_mask   = ~uncertain_mask
    target_names   = list(target_names)

    # ------------------------------------------------------------------
    # 1. Text metrics log
    # ------------------------------------------------------------------
    log_path = OUTPUT_DIR / f"{run_tag}_metrics_{timestamp}.txt"

    lines = [
        "=" * 60,
        "MC Dropout Neural Network — Uncertainty-Aware Inference",
        f"Run Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run Tag  : {run_tag}",
        "=" * 60,
        "",
        f"Total test samples   : {len(y_true)}",
        f"Uncertain (var>{uncertainty_threshold}): "
        f"{uncertain_mask.sum()} ({uncertain_mask.mean()*100:.1f}%)",
        "",
        "--- All Predictions ---",
        classification_report(y_true, y_pred, target_names=target_names),
    ]

    if certain_mask.sum() > 0:
        certain_acc = accuracy_score(y_true[certain_mask], y_pred[certain_mask])
        lines += [
            f"--- Certain Predictions ({certain_mask.sum()} samples) ---",
            f"Accuracy (certain only): {certain_acc * 100:.2f}%",
            classification_report(y_true[certain_mask], y_pred[certain_mask],
                                  target_names=target_names),
        ]

    if uncertain_mask.sum() > 0:
        uncertain_acc = accuracy_score(y_true[uncertain_mask], y_pred[uncertain_mask])
        lines += [
            f"--- Uncertain Predictions ({uncertain_mask.sum()} samples) ---",
            f"Accuracy (uncertain only): {uncertain_acc * 100:.2f}%",
            classification_report(y_true[uncertain_mask], y_pred[uncertain_mask],
                                  target_names=target_names),
        ]

    lines += [
        "--- Uncertainty Score Statistics ---",
        f"  Mean variance : {variance.mean():.5f}",
        f"  Std  variance : {variance.std():.5f}",
        f"  Max  variance : {variance.max():.5f}",
        f"  Min  variance : {variance.min():.5f}",
        "",
        "--- Training History (final epoch) ---",
        f"  Final train loss : {history['train_loss'][-1]:.4f}",
        f"  Final val loss   : {history['val_loss'][-1]:.4f}",
        f"  Final val acc    : {history['val_acc'][-1]*100:.2f}%",
        f"  Best  val loss   : {min(history['val_loss']):.4f}  "
        f"(epoch {history['val_loss'].index(min(history['val_loss']))+1})",
        "=" * 60,
    ]

    with open(log_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"SUCCESS: Metrics log saved     → {log_path}")

    # ------------------------------------------------------------------
    # 2. Learning curves
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss",  color="steelblue")
    axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",    color="tomato")
    axes[0].set_title("BCE Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, [a * 100 for a in history["val_acc"]],
                 color="seagreen", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"MC Dropout — Learning Curves ({run_tag})", fontsize=13, y=1.01)
    plt.tight_layout()
    curves_path = OUTPUT_DIR / f"{run_tag}_learning_curves_{timestamp}.png"
    plt.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SUCCESS: Learning curves saved → {curves_path}")

    # ------------------------------------------------------------------
    # 3. Confusion matrix heatmap  (matches baseline style)
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=target_names, yticklabels=target_names, cbar=False)
    ax.set_title(f"MC Dropout — Confusion Matrix\n(all test samples)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / f"{run_tag}_confusion_matrix_{timestamp}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SUCCESS: Confusion matrix saved → {cm_path}")

    # ------------------------------------------------------------------
    # 4. Uncertainty distribution plots (3-panel)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A — variance histogram split by true class
    for cls_idx, cls_name in enumerate(target_names):
        mask = y_true == cls_idx
        axes[0].hist(variance[mask], bins=40, alpha=0.6, label=cls_name, density=True)
    axes[0].axvline(uncertainty_threshold, color="red", linestyle="--",
                    linewidth=1.5, label=f"Threshold ({uncertainty_threshold})")
    axes[0].set_title("Predictive Variance Distribution\n(by true class)")
    axes[0].set_xlabel("Variance (epistemic uncertainty)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel B — mean probability vs. variance scatter
    colors = np.where(uncertain_mask, "tomato", "steelblue")
    scatter = axes[1].scatter(mean_prob, variance, c=colors, alpha=0.5, s=15)
    axes[1].axhline(uncertainty_threshold, color="red", linestyle="--",
                    linewidth=1.5, label=f"Threshold ({uncertainty_threshold})")
    axes[1].set_title("Mean Probability vs. Variance\n(red = uncertain)")
    axes[1].set_xlabel("Mean Predictive Probability P(Tumor)")
    axes[1].set_ylabel("Predictive Variance")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Panel C — accuracy on certain vs. uncertain subsets
    subset_labels, subset_accs = [], []
    if certain_mask.sum() > 0:
        subset_labels.append(f"Certain\n(n={certain_mask.sum()})")
        subset_accs.append(accuracy_score(y_true[certain_mask], y_pred[certain_mask]) * 100)
    if uncertain_mask.sum() > 0:
        subset_labels.append(f"Uncertain\n(n={uncertain_mask.sum()})")
        subset_accs.append(accuracy_score(y_true[uncertain_mask], y_pred[uncertain_mask]) * 100)
    subset_labels.append(f"All\n(n={len(y_true)})")
    subset_accs.append(accuracy_score(y_true, y_pred) * 100)

    bar_colors = ["steelblue"] * (len(subset_labels) - 1) + ["grey"]
    bars = axes[2].bar(subset_labels, subset_accs, color=bar_colors, edgecolor="white", width=0.5)
    for bar, acc in zip(bars, subset_accs):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)
    axes[2].set_ylim(0, 115)
    axes[2].set_title("Accuracy by Uncertainty Subset")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle(f"MC Dropout — Epistemic Uncertainty Analysis ({run_tag})", fontsize=13, y=1.01)
    plt.tight_layout()
    unc_path = OUTPUT_DIR / f"{run_tag}_uncertainty_{timestamp}.png"
    plt.savefig(unc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SUCCESS: Uncertainty plots saved → {unc_path}")


# ---------------------------------------------------------------------------
# 7. Persistence Utilities
# ---------------------------------------------------------------------------

def save_model(model: MCDropoutClassifier, path: str = None, run_tag: str = "mc_dropout") -> str:
    """
    Save model weights (state_dict) to disk.

    If `path` is not provided the checkpoint is written to OUTPUT_DIR with a
    timestamped filename so every training run produces a traceable artefact.

    Returns the resolved path string.
    """
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(OUTPUT_DIR / f"{run_tag}_model_{timestamp}.pt")
    torch.save(model.state_dict(), path)
    print(f"SUCCESS: Model checkpoint saved → {path}")
    return path


def load_model(path: str, **model_kwargs) -> MCDropoutClassifier:
    """
    Restore a saved MCDropoutClassifier from a state_dict checkpoint.

    Args:
        path        : Path to the .pt / .pth checkpoint file.
        **model_kwargs: Keyword arguments forwarded to MCDropoutClassifier
                        (e.g., input_dim, hidden_dims, dropout_p).
    """
    model = MCDropoutClassifier(**model_kwargs)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f"Model loaded ← {path}")
    return model


# ---------------------------------------------------------------------------
# 7. Quick Smoke-Test (run: python -m models.uncertainty)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    print(f"Device: {DEVICE}")

    # ------------------------------------------------------------------
    # Synthetic data mimicking the real (2700, 604) dataset
    # ------------------------------------------------------------------
    N, D = 2700, 604
    rng = np.random.default_rng(42)
    X_synth = rng.standard_normal((N, D)).astype(np.float32)
    y_synth = (rng.random(N) > 0.5).astype(np.float32)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_synth, y_synth, test_size=0.3, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    # ------------------------------------------------------------------
    # Instantiate model
    # ------------------------------------------------------------------
    model = MCDropoutClassifier(input_dim=D, hidden_dims=(512, 256, 128, 64), dropout_p=0.3)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = train_model(
        model, X_tr, y_tr, X_val, y_val,
        epochs=50, batch_size=64, lr=1e-3, patience=10, verbose=True,
    )

    # ------------------------------------------------------------------
    # MC Dropout Inference (T=50 stochastic passes)
    # ------------------------------------------------------------------
    print("\nRunning MC Dropout inference (T=50)...")
    results = mc_predict(model, X_te, T=50, batch_size=128)

    evaluate_with_uncertainty(results, y_te.astype(int), uncertainty_threshold=0.05)

    # ------------------------------------------------------------------
    # Save all artefacts to OUTPUT_DIR
    # ------------------------------------------------------------------
    save_outputs(
        results=results,
        y_true=y_te.astype(int),
        history=history,
        uncertainty_threshold=0.05,
        target_names=("Reference", "Tumor"),
        run_tag="mc_dropout_smoketest",
    )
    save_model(model, run_tag="mc_dropout_smoketest")
    print("\nSmoke-test complete. Check output/ folder for all artefacts.")
