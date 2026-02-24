from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


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


def build_baseline_pipeline(schema: FeatureSchema, seed:int) -> Pipeline:
    final_pipe = Pipeline([
        ('preprocess',build_preprocessor(schema)),
        ('model',LogisticRegression(max_iter=1000, random_state=seed))
    ])

    return final_pipe
