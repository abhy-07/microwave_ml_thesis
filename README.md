# Uncertainty-Aware Machine Learning for Microwave-Based Body Composition Assessment

**M.Sc. Thesis — Uppsala University, 2026**  
**Author:** Abhishek Yadav  
**Supervisor:** Bappaditya Mandal  
**Subject Reviewer:** Robin Augustine  

---

## Overview

This project develops a **probabilistic machine learning system** that predicts human body composition (skin thickness, subcutaneous fat, and skeletal muscle area) from microwave S-parameter measurements — non-invasively, without ultrasound or X-ray equipment.

The central contribution is **calibrated uncertainty quantification**: instead of a bare point estimate, every prediction comes with a confidence interval that reliably reflects the model's actual accuracy. A clinician who receives `Fat = 9.2 ± 8.8 mm` can immediately decide whether to trust the result or confirm with ultrasound. A system that only outputs `Fat = 11.3 mm` cannot support that decision.

### Clinical Problem

| Method | Cost | Uncertainty? | Portable? |
|---|---|---|---|
| Diagnostic ultrasound (gold standard) | €10k–€100k | No | No |
| DEXA | High + radiation | No | No |
| **This system (microwave VNA)** | **~€50–100** | **Yes** | **Yes** |

---

## Key Results

Evaluated on **471 samples from 23 independent volunteers** (March 2023 cohort, held out completely from training):

### Prediction Accuracy

| Target | Best Model | RMSE | MAE |
|---|---|---|---|
| Skin thickness | XGBoost | **0.382 mm** | 0.298 mm |
| Subcutaneous fat | Random Forest | **4.473 mm** | 3.619 mm |
| Muscle CSA | Random Forest | **2.111 cm²** | 1.602 cm² |

### Uncertainty Calibration (MC Dropout — main contribution)

| Target | ECE ↓ | PICP 95% ↑ | NLL ↓ |
|---|---|---|---|
| Skin | **0.074** | 0.854 | 0.930 |
| Fat | **0.094** | 1.000 | 3.310 |
| Muscle | **0.020** | 0.926 | 2.250 |

> ECE = Expected Calibration Error (0 = perfect). Deterministic baselines (RF, XGBoost, FCNN) achieve PICP = 0.000 and NLL > 10,000 — confirming they provide **no usable probabilistic output**.

### Clinical Risk Score

Using predictive uncertainty (σ_total) as a per-prediction quality flag:

| Target | Flagging Rate | Low-Risk RMSE | High-Risk RMSE | Separation Ratio ρ |
|---|---|---|---|---|
| Skin | 25% | 0.575 mm | 0.609 mm | **1.060×** |
| Fat | 25% | 5.653 mm | 5.828 mm | **1.031×** |
| Muscle | 25% | 2.183 cm² | 2.239 cm² | **1.026×** |

ρ > 1.0 confirms that predictions flagged as uncertain are genuinely less accurate — the model knows when it does not know.

---

## Methods

### Dataset
- **Training:** MAS Volunteer Study, September 2022 — 16 volunteers, 431 S2P files
- **Test:** MAS Volunteer Study, March 2023 — 23 volunteers, 471 S2P files (zero overlap)
- **Sensor:** nanoVNA (~€50), measuring S11 (reflection) and S21 (transmission) at 1–3 GHz
- **Ground truth:** Diagnostic ultrasound (skin, fat, muscle cross-sectional area)

### Feature Engineering
- Parse `.s2p` Touchstone files → extract S11 and S21 (real + imaginary)
- **Neural networks:** subsample to 200 frequency points → **800-dimensional feature vector**
- **Tree models:** 10-band statistics (mean, std, min, max) → **160-dimensional feature vector**
- StandardScaler fitted on training set only; applied to test set without refitting

### Models

```
┌─────────────────────────────────────────────────────────────────┐
│  DETERMINISTIC BASELINES          PROBABILISTIC MODELS          │
│  ┌──────────────┐                 ┌──────────────────────────┐  │
│  │ Random Forest│                 │   MC Dropout (T=50)      │  │
│  │   400 trees  │                 │   Shared encoder +       │  │
│  └──────────────┘                 │   3 heteroscedastic heads│  │
│  ┌──────────────┐                 │   NLL loss: learns μ, σ² │  │
│  │   XGBoost    │                 └──────────────────────────┘  │
│  │  800 trees   │                 ┌──────────────────────────┐  │
│  └──────────────┘                 │   Deep Ensemble (N=5)    │  │
│  ┌──────────────┐                 │   5 independent members  │  │
│  │  Det. FCNN   │                 │   Law of total variance  │  │
│  │ 800→256→128  │                 │   aggregation            │  │
│  │   →64→3      │                 └──────────────────────────┘  │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

**MC Dropout** (Gal & Ghahramani, 2016): keeps dropout active at inference time. T=50 stochastic passes decompose uncertainty into epistemic (model) and aleatoric (data noise) components via the law of total variance.

**Deep Ensembles** (Lakshminarayanan et al., 2017): 5 networks trained from different random seeds. Mixture-of-Gaussians aggregation captures functional diversity.

Both models use a **heteroscedastic Gaussian NLL loss** that forces the network to simultaneously learn accurate predictions and calibrated confidence.

### Evaluation
- **Cross-validation:** 4-fold volunteer-level GroupKFold (prevents label leakage — all files from one volunteer go to the same fold)
- **Calibration metrics:** ECE, PICP, MPIW, NLL, reliability diagrams
- **Clinical risk score:** σ_total > 75th percentile → flag for ultrasound confirmation; measure RMSE separation across strata

---

## Repository Structure

```
microwave_ml_thesis/
│
├── src/                          # Main pipeline (final, production-quality code)
│   ├── config.py                 # Paths, hyperparameters, random seeds
│   ├── run_pipeline.py           # Stage orchestrator — entry point
│   ├── data/
│   │   ├── loader.py             # S2P parser, Excel label merger, quality filtering
│   │   └── features.py           # Band-statistical feature engineering
│   ├── models/
│   │   ├── baselines.py          # Random Forest, XGBoost, deterministic FCNN
│   │   ├── mc_dropout.py         # MC Dropout: architecture, NLL loss, inference, plots
│   │   └── deep_ensemble.py      # Deep Ensemble: training, aggregation, plots
│   └── evaluation/
│       ├── metrics.py            # RMSE, MAE, R², ECE, PICP, MPIW, NLL
│       ├── plots.py              # Scatter plots, reliability diagrams, feature importance
│       ├── plot_style.py         # Shared publication-quality matplotlib settings
│       ├── cross_val.py          # GroupKFold cross-validation for baselines
│       └── risk_score.py         # Risk stratification, threshold sweep, report
│
├── latex/
│   ├── thesis.tex                # Complete thesis (LaTeX source, Overleaf-compatible)
│   └── figures/                  # All 33 figures embedded in the thesis (PNG, 300 DPI)
│
├── outputs/
│   ├── figures/                  # All figures generated by the pipeline
│   ├── metrics/                  # Text result reports (cross-validation, validation, risk)
│   └── models/                   # Saved model weights (.pt) and scaler (.pkl)
│
└── code/                         # Early-stage exploratory code (legacy, see src/ for final)
```

> **Note:** Raw S2P data files (25 GB) are not included in this repository as they contain identifiable volunteer information. The `Datafiles/` directory is excluded via `.gitignore`.

---

## Running the Pipeline

### Requirements

```bash
pip install numpy pandas scikit-learn xgboost torch scipy matplotlib openpyxl scikit-rf
```

Python 3.9+, PyTorch 2.x.

### Steps

```bash
# 1. Set data paths in src/config.py to point to your local Datafiles/ directory

# 2. Run stages individually or all at once
cd src/

python run_pipeline.py --stage data        # Parse S2P files, extract features
python run_pipeline.py --stage baseline    # Train RF, XGBoost, FCNN
python run_pipeline.py --stage cv          # 4-fold GroupKFold cross-validation
python run_pipeline.py --stage mc_dropout  # Train MC Dropout, generate uncertainty plots
python run_pipeline.py --stage ensemble    # Train Deep Ensemble (5 members)
python run_pipeline.py --stage validate    # Evaluate all models on March 2023 test set
python run_pipeline.py --stage risk_score  # Clinical risk stratification analysis

# Or run everything in sequence
python run_pipeline.py --stage all
```

All figures are saved to `outputs/figures/` at 300 DPI. All metric reports are saved to `outputs/metrics/`.

---

## Selected Figures

| Figure | Description |
|---|---|
| `outputs/figures/mc_dropout_learning_curves_*.png` | NLL training and validation curves with best-epoch marker |
| `outputs/figures/mc_dropout_uncertainty_Fat_mm_*.png` | 3-panel: predictions vs truth, epistemic vs aleatoric decomposition, σ histogram |
| `outputs/figures/val_mc_*_*.png` | Predicted vs ground truth with ±2σ error bars on the independent test set |
| `latex/figures/rel_mc_*.png` | Reliability (calibration) diagrams — model curve vs perfect-calibration diagonal |
| `latex/figures/risk_mc_*.png` | Risk stratification: scatter by tier, \|error\| vs σ, RMSE bar chart |
| `latex/figures/risk_sweep_mc_*.png` | Threshold sweep: RMSE per stratum and separation ratio vs flagging rate |

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![matplotlib](https://img.shields.io/badge/matplotlib-3.x-blue)

**Core stack:** Python · PyTorch · scikit-learn · XGBoost · NumPy · pandas · SciPy · matplotlib  
**Domain tools:** scikit-rf (S-parameter parsing) · openpyxl (label extraction)  
**Thesis:** LaTeX (Overleaf-compatible, pdflatex only)

---

## References

1. Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML.
2. Lakshminarayanan, B. et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS.
3. Kendall, A. & Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS.
4. Mattsson, V. et al. (2022). *Machine learning for non-invasive microwave assessment of body composition*. Sensors.

---

*Uppsala University, Department of Electrical Engineering, 2026*
