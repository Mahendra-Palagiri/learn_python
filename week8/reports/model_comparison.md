# Model Comparison (Week 8 Capstone)

## Baseline — LogisticRegression (Leakage-Safe Pipeline) - Phase 2

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

## Candidates — Fair CV Comparison (Same Protocol) - Phase 3

- **Dataset:** Seaborn Titanic (cached)
- **Target:** survived
- **Leakage Prevention:** dropped `alive`
- **Split:** Holdout test = 20% (not used in CV)
- **CV:** StratifiedKFold (5 folds, shuffle=True, seed=42)
- **Primary Metric:** ROC-AUC

### Results (ROC-AUC mean ± std)
- LogisticRegression: 0.8639 ± 0.0215
  - folds: [0.8657, 0.8294, 0.8908, 0.8530, 0.8805]
- RandomForest: 0.8626 ± 0.0132
  - folds: [0.8620, 0.8433, 0.8724, 0.8809, 0.8546]
- GradientBoosting: 0.8867 ± 0.0235
  - folds: [0.8952, 0.8428, 0.9120, 0.8975, 0.8859]

### Provisional Decision (Phase 3)
- We select **GradientBoosting** as the provisional leader (highest mean ROC-AUC).
- We note RandomForest is the most stable (lowest std), but its mean ROC-AUC is lower.
- We proceed to Phase 4 tuning for GradientBoosting using CV only (no test-set access).