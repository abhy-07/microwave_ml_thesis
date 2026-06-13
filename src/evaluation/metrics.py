"""
metrics.py
==========
Uncertainty-aware evaluation metrics for regression models.

Key metrics implemented:
  - RMSE, MAE, R²                 (prediction accuracy)
  - Expected Calibration Error    (ECE — are confidence intervals reliable?)
  - Prediction Interval Coverage  (PICP — do intervals contain the true value?)
  - Mean Prediction Interval Width (MPIW — are intervals tight?)
  - Sharpness                     (average predictive std)

References:
  - Gneiting & Raftery (2007). Strictly proper scoring rules, prediction, and estimation.
  - Kuleshov et al. (2018). Accurate uncertainties for deep learning using calibration.
"""

import numpy as np
import pandas as pd
from typing import Optional


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def picp(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Prediction Interval Coverage Probability.
    Fraction of true values that fall within [y_lower, y_upper].
    Ideal: equals the nominal coverage level (e.g. 0.95 for 95% PI).
    """
    covered = np.logical_and(y_true >= y_lower, y_true <= y_upper)
    return float(np.mean(covered))


def mpiw(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Mean Prediction Interval Width. Smaller is better (given sufficient coverage)."""
    return float(np.mean(y_upper - y_lower))


def sharpness(std: np.ndarray) -> float:
    """Average predictive standard deviation. Smaller = more confident."""
    return float(np.mean(std))


def expected_calibration_error(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std:  np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error for regression (Gaussian predictive distribution).

    For each confidence level α ∈ {0.1, 0.2, ..., 1.0}, compute:
      - Expected coverage: α
      - Observed coverage: fraction of y_true within the α-PI
    ECE = mean |expected_coverage - observed_coverage| across all bins.

    Args:
        y_true : (n,) ground truth
        y_mean : (n,) predictive mean
        y_std  : (n,) predictive std
        n_bins : number of confidence levels to evaluate

    Returns:
        ECE (scalar, lower is better; 0 = perfectly calibrated)
    """
    from scipy import stats

    alphas = np.linspace(0.1, 1.0, n_bins)
    errors = []

    for alpha in alphas:
        z = stats.norm.ppf((1 + alpha) / 2)  # two-sided z-score
        lower = y_mean - z * y_std
        upper = y_mean + z * y_std
        observed = np.mean((y_true >= lower) & (y_true <= upper))
        errors.append(abs(alpha - observed))

    return float(np.mean(errors))


def nll_gaussian(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> float:
    """
    Negative Log-Likelihood under a Gaussian predictive distribution.
    Lower is better.
    """
    var = y_std ** 2 + 1e-6
    nll = 0.5 * np.mean(np.log(2 * np.pi * var) + (y_true - y_mean) ** 2 / var)
    return float(nll)


def full_uncertainty_report(
    y_true:  np.ndarray,
    y_mean:  np.ndarray,
    y_std:   np.ndarray,
    target_name: str = "target",
    coverage: float = 0.95,
) -> pd.Series:
    """
    Compute all uncertainty metrics for one target variable and return as a Series.

    Args:
        y_true      : (n,) ground truth values
        y_mean      : (n,) predictive means
        y_std       : (n,) predictive stds
        target_name : label for the Series
        coverage    : nominal coverage for PICP (default 0.95)
    """
    from scipy import stats

    z = stats.norm.ppf((1 + coverage) / 2)
    y_lower = y_mean - z * y_std
    y_upper = y_mean + z * y_std

    return pd.Series(
        {
            "RMSE":        rmse(y_true, y_mean),
            "MAE":         mae(y_true, y_mean),
            "R2":          r2(y_true, y_mean),
            f"PICP_{int(coverage*100)}": picp(y_true, y_lower, y_upper),
            f"MPIW_{int(coverage*100)}": mpiw(y_lower, y_upper),
            "Sharpness":   sharpness(y_std),
            "ECE":         expected_calibration_error(y_true, y_mean, y_std),
            "NLL":         nll_gaussian(y_true, y_mean, y_std),
        },
        name=target_name,
    )
