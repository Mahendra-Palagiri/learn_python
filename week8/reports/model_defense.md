# Week 8 Mini Capstone — Model Defense

## Evaluation Rules (Locked)
- We keep a **final holdout test set** and evaluate on it **once**.
- We perform all preprocessing **inside a sklearn Pipeline**.
- We compare models using the **same CV strategy** and the **same primary metric**.
- We select models based on **CV mean + variance** (stability matters).
- We fix and record all random seeds.

## Phase 5 — Final Holdout Test (Single Use)

- **Dataset:** Seaborn Titanic (cached)
- **Target:** survived
- **Leakage Prevention:** dropped `alive`
- **Split:** Holdout test = 20% (seed=42, stratified)
- **Model:** GradientBoosting (tuned)

### Tuned Params
- subsample: 1.0
- n_estimators: 350
- min_samples_leaf: 10
- max_depth: 2
- learning_rate: 0.18

### Holdout Metrics (Test Set)
- roc_auc: 0.8079
- accuracy: 0.7765
- precision: 0.7302
- recall: 0.6667
- f1: 0.6970

### Confusion Matrix (threshold=0.5)
[[93, 17],
 [23, 46]]