from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


@dataclass(frozen=True)
class FeatureSchema:
    numeric: list[str]
    categorical: list[str]


def infer_basic_schema(X: pd.DataFrame) -> FeatureSchema:
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return FeatureSchema(numeric=numeric, categorical=categorical)


def build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:

    num_pipe = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore'))
    ])

    pre_prcsr = ColumnTransformer([
        ('num',num_pipe,schema.numeric),
        ('cat',cat_pipe,schema.categorical)
    ],remainder='drop')

    return pre_prcsr

def build_logreg(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=2000,random_state=seed)

def build_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=400,random_state=seed,n_jobs=-1)

#Phase-3
# def build_gb(seed: int) -> GradientBoostingClassifier:
#     return GradientBoostingClassifier(random_state=seed)

#Phase-4
def build_gb(seed: int,**params) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=seed,**params)

def build_pipeline(model, schema: FeatureSchema) -> Pipeline:
    final_pipe = Pipeline([
        ('preprocess',build_preprocessor(schema)),
        ('model',model)
    ])

    return final_pipe
