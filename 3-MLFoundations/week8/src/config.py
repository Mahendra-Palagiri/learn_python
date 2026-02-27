from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.2
    n_splits_cv: int = 5

    # Classification metrics (primary drives selection)
    primary_scoring: str = "roc_auc"
    secondary_scoring: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")

CFG = Config()