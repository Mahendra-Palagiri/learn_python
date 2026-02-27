from __future__ import annotations

from sklearn.model_selection import train_test_split

from config import CFG
from data_load import DatasetSpec,load_dataset
from pipeline import infer_basic_schema #This is our pipeline not from sklearn schematics.
from pipeline import build_logreg,build_rf,build_gb,build_pipeline
from evaluate import run_cv



def main() -> None:

    spec = DatasetSpec(
        source="seaborn",
        filename="titanic_seaborn_cached.csv",
        target_col="survived",
        leakage_cols=['alive'], #We have added this after Phase 2 evaluation becuase this is giving out the answer
        seaborn_name="titanic",
    )

    titdf = load_dataset(spec).dropna(axis=1, how='all')

    # print(titdf.info())
    # print(titdf.describe(include='all'))

    X = titdf.drop(columns=[spec.target_col])
    X = X.drop(columns=[c for c in spec.leakage_cols if c in X.columns]) #Drop the leakage columns
    Y = titdf[spec.target_col]

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=CFG.test_size,random_state=CFG.seed,stratify=Y)
    schema = infer_basic_schema(X_train)

    # print("\nPhase 1 — Data & Split Design")
    # print("Dataset:", spec.filename, "| target:", spec.target_col)
    # print("\nShapes")
    # print("X_train:", X_train.shape, "y_train:", Y_train.shape)
    # print("X_test :", X_test.shape, "y_test :", Y_test.shape)

    # print("\nTarget Distribution (train)")
    # print(Y_train.value_counts(normalize=True).round(3))

    # print("\nTarget Distribution (test)")
    # print(Y_test.value_counts(normalize=True).round(3))

    # print("\nSchema")
    # print("Numeric cols     :", schema.numeric)
    # print("Categorical cols :", schema.categorical)

    # print("\nSanity Checks")
    # print("- Cached file exists:", (titdf is not None))

    # =========== Phase-2 =======
    # pipe = build_baseline_pipeline(schema=schema,seed=CFG.seed)

    # summary = run_cv(
    #     estimator=pipe,
    #     X=X_train,
    #     Y=Y_train,
    #     seed=CFG.seed,
    #     n_splits=CFG.n_splits_cv,
    #     primary_scoring=CFG.primary_scoring,
    #     secondary_scoring=CFG.secondary_scoring
    # )

    # print("\nPhase 2 — Baseline (LogisticRegression) CV Summary")
    # print(f"Primary metric: {summary.primary}")
    # print(f"CV mean: {summary.mean:.4f}  |  CV std: {summary.std:.4f}")
    # print(f"Fold scores: {[round(v, 4) for v in summary.folds]}")

    # print("\nAll metrics (mean ± std)")
    # for m in (CFG.primary_scoring,) + CFG.secondary_scoring:
    #     print(f"- {m:>10}: {summary.metrics_mean[m]:.4f} ± {summary.metrics_std[m]:.4f}")

    # =========== Phase-3 =======
    candidates= [
        ("LogisitcRegression",build_pipeline(build_logreg(CFG.seed),schema=schema)),
        ("RandomForest",build_pipeline(build_rf(CFG.seed),schema=schema)),
        ("GradientBoosting",build_pipeline(build_gb(CFG.seed),schema=schema))
    ]

    print("\nPhase 3 — Candidate Model Comparison (CV on Training Only)")
    print(f"Primary metric: {CFG.primary_scoring}")
    print(f"CV: StratifiedKFold (n_splits={CFG.n_splits_cv}, seed={CFG.seed})")

    results =[]
    for name,pipe in candidates:
        summary = run_cv(
            estimator=pipe,
            X=X_train,
            Y=Y_train,
            seed=CFG.seed,
            n_splits=CFG.n_splits_cv,
            primary_scoring=CFG.primary_scoring,
            secondary_scoring=CFG.secondary_scoring
        )

        results.append((name,summary))
        print(f"\n{name}")
        print(f"- {summary.primary} mean ± std: {summary.mean:.4f} ± {summary.std:.4f}")
        print(f"- folds: {[round(v, 4) for v in summary.folds]}")


    # Quick ranking by primary metric mean
    results_sorted = sorted(results, key=lambda x: x[1].mean, reverse=True)

    print("\nRanking (by primary metric mean)")
    for i, (name, s) in enumerate(results_sorted, start=1):
        print(f"{i}. {name:>16} | {s.mean:.4f} ± {s.std:.4f}")




if __name__ =='__main__':
    main()