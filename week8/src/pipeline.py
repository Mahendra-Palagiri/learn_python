from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class FeatureSchema:
    numeric: list[str]
    categorical: list[str]


def infer_basic_schema(X: pd.DataFrame) -> FeatureSchema:
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return FeatureSchema(numeric=numeric, categorical=categorical)