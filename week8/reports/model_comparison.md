# Model Comparison (Week 8 Capstone)

## Baseline — LogisticRegression (Leakage-Safe Pipeline)

- **Dataset:** Seaborn Titanic (cached)
- **Target:** survived
- **Leakage Prevention:** dropped `alive`
- **Split:** Holdout test = 20% (not used in CV)
- **CV:** StratifiedKFold (5 folds, shuffle=True, seed=42)
- **Primary Metric:** ROC-AUC

### CV Results
- **ROC-AUC (mean ± std):** 0.8639 ± 0.0215
- **ROC-AUC fold scores:** [0.8657, 0.8294, 0.8908, 0.8530, 0.8805]

### Secondary Metrics (mean ± std)
- accuracy: 0.8175 ± 0.0334
- precision: 0.7731 ± 0.0690
- recall: 0.7547 ± 0.0401
- f1: 0.7613 ± 0.0351