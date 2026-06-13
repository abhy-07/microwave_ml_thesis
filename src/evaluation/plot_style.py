"""
plot_style.py
=============
Centralised matplotlib style settings for publication-quality figures.

Call apply_pub_style() at the top of any plotting module to get consistent,
high-resolution, large-font figures suitable for journal submission or
thesis printing.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_pub_style() -> None:
    """Apply publication-quality rcParams globally."""
    plt.rcParams.update({
        # --- Font sizes ---
        "font.size":          14,
        "axes.titlesize":     17,
        "axes.labelsize":     15,
        "xtick.labelsize":    13,
        "ytick.labelsize":    13,
        "legend.fontsize":    13,
        "figure.titlesize":   19,

        # --- Font weight ---
        "axes.titleweight":   "bold",
        "figure.titleweight": "bold",

        # --- Lines ---
        "lines.linewidth":    2.5,
        "lines.markersize":   8,

        # --- Axes ---
        "axes.linewidth":     1.4,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linewidth":     0.8,
        "grid.linestyle":     "--",

        # --- Ticks ---
        "xtick.major.width":  1.2,
        "ytick.major.width":  1.2,
        "xtick.major.size":   5,
        "ytick.major.size":   5,
        "xtick.direction":    "out",
        "ytick.direction":    "out",

        # --- Figure ---
        "figure.dpi":         150,          # screen preview
        "savefig.dpi":        300,          # saved to disk
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        "figure.constrained_layout.use": True,

        # --- Legend ---
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "0.8",
        "legend.borderpad":   0.6,

        # --- Colour cycle (high-contrast) ---
        "axes.prop_cycle": matplotlib.cycler(
            color=["#2166AC", "#D6604D", "#4DAC26", "#8073AC",
                   "#E08214", "#01665E", "#762A83", "#1A1A1A"]
        ),
    })


# Shared colour palette for consistent cross-figure identity
PALETTE = {
    "low_risk":   "#2166AC",   # strong blue
    "high_risk":  "#D6604D",   # strong red-orange
    "mc":         "#2166AC",
    "ensemble":   "#D6604D",
    "rf":         "#4DAC26",
    "xgb":        "#8073AC",
    "fcnn":       "#E08214",
    "perfect":    "#1A1A1A",
    "threshold":  "#555555",
    "trend":      "#333333",
}

# Standard units per target (used in axis labels)
TARGET_UNITS = {
    "Skin_mm":    "mm",
    "Fat_mm":     "mm",
    "Muscle_cm2": "cm²",
}

TARGET_DISPLAY = {
    "Skin_mm":    "Skin Thickness",
    "Fat_mm":     "Subcutaneous Fat Thickness",
    "Muscle_cm2": "Rectus Femoris CSA",
}
