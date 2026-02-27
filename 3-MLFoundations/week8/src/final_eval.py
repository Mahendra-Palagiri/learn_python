#Phase 5
from __future__ import annotations

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import CFG
from data_load import DatasetSpec, load_dataset
from pipeline import infer_basic_schema, build_pipeline, build_gb


def main() -> None:
    spec = DatasetSpec(
        source="seaborn",
        filename="titanic_seaborn_cached.csv",
        target_col="survived",
        leakage_cols=['alive'], 
        seaborn_name="titanic",
    )

    df = load_dataset(spec).dropna(axis=1, how="all")

    X = df.drop(columns=[spec.target_col])
    X = X.drop(columns=[c for c in spec.leakage_cols if c in X.columns]) #Drop the leakage columns

    y = df[spec.target_col]

    # Sacred holdout test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CFG.test_size,
        random_state=CFG.seed,
        stratify=y,
    )

    schema = infer_basic_schema(X_train)

    # Best tuned params from Phase 4
    best_model = build_gb(
        CFG.seed,
        subsample=1.0,
        n_estimators=350,
        min_samples_leaf=10,
        max_depth=2,
        learning_rate=0.18,
    )

    pipe = build_pipeline(best_model, schema)

    # Fit on full training data, evaluate once on test
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }

    print("\nPhase 5 — Final Holdout Test (Single Use)")
    for k, v in metrics.items():
        print(f"- {k:>10}: {v:.4f}")

    print("\nConfusion Matrix (threshold=0.5)")
    print(confusion_matrix(y_test, pred))


if __name__ == "__main__":
    main()