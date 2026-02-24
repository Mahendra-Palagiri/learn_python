'''
Data loading utilities for the AI Journey week 8 capstone project.

This module provides functionality to load datasets from local files or seaborn's 
built-in datasets, with automatic caching support.

Classes:
    DatasetSpec: A frozen dataclass that specifies dataset source and metadata.
        - source: Either "local" (for CSV files) or "seaborn" (for seaborn datasets)
        - filename: Name of the cached CSV file in the data directory
        - target_col: The name of the target column for ML models
        - seaborn_name: Optional name of the seaborn dataset (required if source="seaborn")

Functions:
    project_root() -> Path:
        Returns the absolute path to the project root directory (3 levels up from this file).
    
    data_dir() -> Path:
        Returns the path to the data directory: {project_root}/projects/week8_capstone/data
        Creates the directory if it doesn't exist.
    
    load_dataset(spec: DatasetSpec) -> pd.DataFrame:
        Loads a dataset based on the provided specification.
        - For "local" source: Reads from CSV file or raises FileNotFoundError
        - For "seaborn" source: Loads from seaborn, caches to CSV, or reads cached copy
        - Validates that the target column exists in the loaded dataframe
        - Raises ValueError if source is invalid, seaborn_name is missing, or target_col is not found
        
        Returns:
            A pandas DataFrame containing the loaded dataset.
'''

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class DatasetSpec:
    source: str              # "local" or "seaborn"
    filename: str            # local cache filename
    target_col: str
    leakage_cols: list[str]
    seaborn_name: str | None = None  # e.g. "titanic"

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def data_dir() -> Path:
    return project_root() /  "week8" / "data"

def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    path = data_dir() / spec.filename
    data_dir().mkdir(parents=True, exist_ok=True)

    if spec.source == "local":
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)

    elif spec.source == "seaborn":
        # 1) use cached copy if it exists
        if path.exists():
            df = pd.read_csv(path)
        else:
            import seaborn as sns
            if not spec.seaborn_name:
                raise ValueError("seaborn_name must be provided when source='seaborn'")
            df = sns.load_dataset(spec.seaborn_name)
            df.to_csv(path, index=False)  # cache the exact dataset we used

    else:
        raise ValueError("spec.source must be 'local' or 'seaborn'")

    if spec.target_col not in df.columns:
        raise ValueError(f"Target column '{spec.target_col}' not found. Columns: {list(df.columns)}")

    return df