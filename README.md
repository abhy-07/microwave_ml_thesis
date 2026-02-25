# Uncertainty-Aware Machine Learning for Microwave-Based Body Composition Assessment

[cite_start]**M.Sc Computer Science Master Thesis** [cite: 2, 3]  
[cite_start]**University:** Uppsala University [cite: 19]  
[cite_start]**Author:** Abhishek Yadav [cite: 6]  
[cite_start]**Supervisor:** Bappaditya Mandal [cite: 10]  
[cite_start]**Subject Reviewer:** Robin Augustine [cite: 14]  

## 📌 Project Overview
[cite_start]Noninvasive microwave sensing has emerged as a promising technique for biomedical diagnostics[cite: 28]. [cite_start]However, a critical limitation for clinical translation is that current machine learning models typically output point estimates without prediction confidence[cite: 29]. [cite_start]In real clinical settings, measurement noise and biological variability can lead to unreliable predictions[cite: 30]. 

This project aims to bridge the "clinical translation gap" by developing an uncertainty-aware machine learning system. [cite_start]By integrating Bayesian techniques, the model will quantify predictive uncertainty to provide a clinically interpretable risk score, supporting safer medical decision-making[cite: 31, 32].

## 🚀 Current Progress (Weeks 1-6)
- **Data Ingestion Pipeline:** Built a robust `scikit-rf` pipeline to process 2,700 `.s2p` Vector Network Analyzer files, extracting $S_{11}$ (Reflection) and $S_{21}$ (Transmission) frequencies into a structured $(2700, 604)$ feature matrix.
- **Exploratory Data Analysis (EDA):** Visualized mean frequency responses, successfully identifying distinct dielectric shifts in anomalous (tumor) tissues compared to reference tissues.
- [cite_start]**Deterministic Baseline:** Established a Baseline Random Forest Classifier [cite: 64, 65] for binary anomaly detection (Tumor vs. Reference). 
    - **Current Accuracy:** 85.93%
    - **Clinical Insight:** While recall for tumors is extremely high (0.97), the model suffers from false positives due to baseline signal noise. [cite_start]Because this deterministic model cannot quantify its own uncertainty[cite: 88, 89], it strictly validates the need for the upcoming Bayesian models.

## 🧬 Next Steps (Weeks 7-11)
[cite_start]Transitioning from deterministic baselines to Uncertainty-Aware Modeling Approaches[cite: 68, 69]:
1. [cite_start]**Monte Carlo Dropout:** Applying stochastic forward passes to approximate Bayesian uncertainty[cite: 71].
2. [cite_start]**Deep Ensembles:** Training multiple model initializations to estimate uncertainty via prediction diversity[cite: 72].
3. [cite_start]**Bayesian Neural Networks (BNNs):** Treating network weights as probability distributions[cite: 70].

## 📂 Repository Structure
* `/code/data/` - Pipeline scripts for `.s2p` extraction.
* `/code/models/` - Baseline architectures (Random Forest, XGBoost) and upcoming Uncertainty models (BNNs).
* `/code/output/` - Auto-generated high-resolution feature importance and frequency response plots.
* `main.py` - Core execution script.