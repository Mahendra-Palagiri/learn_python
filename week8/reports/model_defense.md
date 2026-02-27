# Week 8 Mini Capstone — Model Defense

## 1) Problem Framing
- **Task Type:** Binary classification
- **Objective:** Predict Titanic survival (`survived`) from passenger attributes.
- **Dataset:** Seaborn Titanic dataset (cached locally for reproducibility).

## 2) Metric Choice
- **Primary Metric:** ROC-AUC  
  - We use ROC-AUC to evaluate ranking quality independent of a fixed threshold.
- **Secondary Metrics:** accuracy, precision, recall, f1  
  - We use these to understand thresholded classification behavior at 0.5.

## 3) Validation Design (Leakage-Safe)
### Data Split Discipline
- We create a **holdout test split (20%)** using a fixed seed and stratification.
- We use the test split **once** for final reporting.

### Cross-Validation Protocol (Model Selection)
- We use **StratifiedKFold (5 folds, shuffle=True, seed=42)** on the training split only.
- We compare candidates using **CV mean + variance** (stability matters).

### Leakage Prevention
- We remove the `alive` column (a direct target leakage feature in Seaborn Titanic).
- We place all preprocessing inside a **sklearn Pipeline** so that:
  - imputation, encoding, and scaling occur within each CV fold
  - no global statistics leak across folds

## 4) Models Evaluated
### Baseline
- **LogisticRegression** with a leakage-safe preprocessing Pipeline.

### Candidate Models (same protocol)
- **RandomForestClassifier**
- **GradientBoostingClassifier**

## 5) Results

### Phase 2 — Baseline (CV on Training Only)
- **LogisticRegression ROC-AUC (mean ± std):** 0.8639 ± 0.0215
- **Secondary metrics (mean ± std):**
  - accuracy: 0.8175 ± 0.0334
  - precision: 0.7731 ± 0.0690
  - recall: 0.7547 ± 0.0401
  - f1: 0.7613 ± 0.0351

### Phase 3 — Candidate Comparison (CV on Training Only)
- **GradientBoosting ROC-AUC (mean ± std):** 0.8867 ± 0.0235
- **LogisticRegression ROC-AUC (mean ± std):** 0.8639 ± 0.0215
- **RandomForest ROC-AUC (mean ± std):** 0.8626 ± 0.0132

**Selection Decision (Phase 3):**
- We select **GradientBoosting** as the provisional leader based on highest mean ROC-AUC under identical CV protocol.

### Phase 4 — Hyperparameter Tuning (CV on Training Only)
- **Method:** RandomizedSearchCV (n_iter=30)
- **Best ROC-AUC (CV mean ± std):** 0.8895 ± 0.0214
- **Best Params:**
  - subsample: 1.0
  - n_estimators: 350
  - min_samples_leaf: 10
  - max_depth: 2
  - learning_rate: 0.18

### Phase 5 — Final Holdout Test (Single Use)
- **Holdout ROC-AUC:** 0.8079
- **Holdout metrics (threshold=0.5):**
  - accuracy: 0.7765
  - precision: 0.7302
  - recall: 0.6667
  - f1: 0.6970
- **Confusion Matrix (threshold=0.5):**
  - [[93, 17],
     [23, 46]]

## 6) Risks and Limitations
- The holdout result is lower than CV estimates, indicating potential optimism from model selection/tuning and/or variance from a single split.
- Default threshold (0.5) may not align with operational goals (precision vs recall trade-off).
- Feature set is limited and includes missing values; additional feature engineering could improve signal.

## 7) Improvement Plan
- **Threshold tuning:** Choose a threshold based on training CV (or an internal validation split), not repeated test probing.
- **Calibration:** Evaluate probability calibration (e.g., Platt scaling or isotonic) if probability quality matters.
- **Feature engineering:** Investigate better handling of cabin/ticket/name fields (carefully, to avoid leakage).
- **Stability checks:** Repeat evaluation with multiple random seeds or repeated CV for more robust estimates.

## 8) Reproducibility Notes
- Fixed seed: 42
- Cached dataset file: `projects/week8_capstone/data/titanic_seaborn_cached.csv`
- Sacred holdout test set used once for final evaluation.