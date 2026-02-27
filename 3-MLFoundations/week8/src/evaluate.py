from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from sklearn.model_selection import StratifiedKFold,cross_validate

@dataclass(frozen=True)
class CVSummary:
    primary: str
    mean: float
    std: float
    folds: list[float]
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]


def run_cv(estimator,X,Y,seed:int,n_splits:int,primary_scoring:str,secondary_scoring:tuple[str,...]) -> CVSummary:

    cv = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    scoring = (primary_scoring,)+(secondary_scoring)
    scores = cross_validate(
        estimator=estimator,
        X=X,
        y=Y,
        scoring=scoring,
        cv=cv,
        return_train_score=False
    )

    primary_key = f'test_{primary_scoring}'
    primary_folds = scores[primary_key]
    mean = float(np.mean(primary_folds))
    std = float(np.std(primary_folds))

    metric_mean: dict[str,float] = {}
    metric_std: dict[str,float] = {}

    for metric in scoring:
        vals = scores[f'test_{metric}']
        metric_mean[metric]= float(np.mean(vals))
        metric_std[metric] = float(np.std(vals))

    return CVSummary(
        primary=primary_scoring,
        mean=mean,
        std=std,
        folds=[float(v) for v in primary_folds],
        metrics_mean=metric_mean,
        metrics_std=metric_std
    )
