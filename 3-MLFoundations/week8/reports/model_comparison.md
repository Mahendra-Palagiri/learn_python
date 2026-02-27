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


## Phase 4 — Tuning (GradientBoosting)

- **Method:** RandomizedSearchCV
- **Budget:** n_iter = 30
- **CV:** StratifiedKFold (5 folds, shuffle=True, seed=42)
- **Primary Metric:** ROC-AUC

### Best Result
- **Best ROC-AUC (CV mean ± std):** 0.8895 ± 0.0214
- **Best Params:**
  - model__subsample: 1.0
  - model__n_estimators: 350
  - model__min_samples_leaf: 10
  - model__max_depth: 2
  - model__learning_rate: 0.18

### Top 5 Configs (rank | mean ± std | params)
1) 0.8895 ± 0.0214 | subsample=1.0, n_estimators=350, min_samples_leaf=10, max_depth=2, learning_rate=0.18  
2) 0.8868 ± 0.0244 | subsample=1.0, n_estimators=350, min_samples_leaf=1,  max_depth=2, learning_rate=0.10  
3) 0.8848 ± 0.0141 | subsample=0.8, n_estimators=250, min_samples_leaf=10, max_depth=3, learning_rate=0.16  
4) 0.8846 ± 0.0248 | subsample=1.0, n_estimators=225, min_samples_leaf=1,  max_depth=2, learning_rate=0.16  
5) 0.8825 ± 0.0271 | subsample=0.8, n_estimators=325, min_samples_leaf=10, max_depth=2, learning_rate=0.02