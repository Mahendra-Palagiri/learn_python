from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV,train_test_split

from data_load import DatasetSpec,load_dataset
from pipeline import infer_basic_schema, build_pipeline,build_gb
from config import CFG


def main() -> None:
   spec = DatasetSpec(
        source="seaborn",
        filename="titanic_seaborn_cached.csv",
        target_col="survived",
        leakage_cols=['alive'], 
        seaborn_name="titanic",
    )
   
   titdf = load_dataset(spec).dropna(axis=1, how='all')

   X = titdf.drop(columns=[spec.target_col])
   X = X.drop(columns=[c for c in spec.leakage_cols if c in X.columns]) #Drop the leakage columns
   Y = titdf[spec.target_col]

   X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=CFG.test_size,random_state=CFG.seed,stratify=Y)
   schema = infer_basic_schema(X_train)

   base_pipe = build_pipeline(build_gb(seed=CFG.seed),schema=schema)

   cv = StratifiedKFold(n_splits=CFG.n_splits_cv,shuffle=True,random_state=CFG.seed)

   param_distributions = {
       "model__n_estimators": np.arange(50,401,25),
       "model__learning_rate": np.linspace(0.02,0.2,10),
       "model__max_depth": [1,2,3],
       "model__subsample": [0,6,0.8,1.0],
       "model__min_samples_leaf": [1,2,5,10]
   }

   search = RandomizedSearchCV(
       estimator=base_pipe,
       param_distributions=param_distributions,
       n_iter=30,  #Fixed number of Random combinations from the parameters provided
       scoring=CFG.primary_scoring,
       n_jobs=-1,
       verbose=1,
       cv=cv,
       random_state=CFG.seed,
       return_train_score=False
   )

   search.fit(X_train, Y_train)


   print("\nPhase 4 — GradientBoosting Tuning (RandomizedSearchCV)")
   print("Best ROC-AUC:", round(search.best_score_, 4))
   print("Best params:")
   for k, v in search.best_params_.items():
      print(f"- {k}: {v}")

   # Top 5 configs
   results = search.cv_results_
   ranks = results["rank_test_score"]
   means = results["mean_test_score"]
   stds = results["std_test_score"]
   params = results["params"]

   top = sorted(
      [(ranks[i], means[i], stds[i], params[i]) for i in range(len(ranks))],
      key=lambda x: x[0]
   )[:5]

   print("\nTop 5 Configs (rank | mean ± std | params)")
   for r, m, s, p in top:
      print(f"{r:>4} | {m:.4f} ± {s:.4f} | {p}")



if __name__ == '__main__':
    main()


#=========================================================
# Retrospection
#=========================================================
'''
Q) Why Random serach and not GridSerach CV

    RandomizedSearchCV is basically “hyperparameter tuning with a budget.”

    Instead of trying every combination in a grid (GridSearchCV), it:
        1.	defines a search space for each hyperparameter, and
        2.	samples a fixed number of random combinations (n_iter),
        3.	evaluates each sampled combo using cross-validation,
        4.	returns the best-performing combo by the scoring metric.

    Why we use it in Phase 4

    Hyperparameter spaces blow up fast. If we used a grid like:
        •	n_estimators: 15 values
        •	learning_rate: 10 values
        •	max_depth: 3 values
        •	subsample: 3 values
        •	min_samples_leaf: 4 values

    Total grid combos = 15 × 10 × 3 × 3 × 4 = 5,400 models
    With 5-fold CV, that’s 27,000 fits. That’s a small campfire that becomes a forest fire.

    RandomizedSearchCV with n_iter=30 is 30 × 5 = 150 fits. Controlled, fast, and usually good enough.

    What “random” really means here
        •	For each iteration, it picks one value from each parameter distribution.
        •	If we provide a list/array (like our np.arange(...) or np.linspace(...)), it samples uniformly from those candidates.
        •	If we provide a distribution object (e.g., from scipy.stats), it samples from that distribution (useful when the parameter scale is not linear).

    Why it often beats GridSearchCV (in practice)

    Not because it’s smarter—because it spends effort more efficiently:
        •	In many models, only a few hyperparameters matter a lot.
        •	Random search explores “more territory” early.
        •	Grid search wastes time on many unimportant combinations.

    There’s a classic result (Bergstra & Bengio, 2012) showing random search can be more efficient than grid search when only a subset of hyperparameters truly drives performance.

    How to interpret the outputs we saw
        •	best_score_: best mean CV score among tried combinations (not test score).
        •	best_params_: the sampled hyperparameter combination that achieved that best mean.
        •	cv_results_: full audit trail:
        •	mean_test_score, std_test_score
        •	rank of each configuration
        •	params for each configuration

    In our run:
        •	we got 0.8895 ± 0.0214 as the best CV ROC-AUC.
        •	And we printed the top 5 mean ± std, which is exactly the right discipline (mean alone is not enough).

    Clinical rules for using RandomizedSearchCV well
        •	Keep n_iter modest (we used 30). Increase only if results are unstable or flat.
        •	Tune a small number of meaningful knobs (we tuned 5).
        •	Use a fixed random_state for reproducibility.
        •	Always tune inside a Pipeline + CV (so preprocessing stays leakage-safe).

    That’s RandomizedSearchCV: sampling-based CV tuning with a fixed compute budget, designed to find strong configs without blowing up runtime.

'''