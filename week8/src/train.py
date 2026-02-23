from __future__ import annotations

from sklearn.model_selection import train_test_split

from config import CFG
from data_load import DatasetSpec,load_dataset
from pipeline import infer_basic_schema #This is our pipeline not sklearn schematics.
import pandas as pd


def main() -> None:

    spec = DatasetSpec(
        source="seaborn",
        filename="titanic_seaborn_cached.csv",
        target_col="survived",
        seaborn_name="titanic",
    )

    titdf = load_dataset(spec).dropna(axis=1, how='all')

    # print(titdf.info())
    # print(titdf.describe(include='all'))

    X = titdf.drop(columns=[spec.target_col])
    Y = titdf[spec.target_col]

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=CFG.test_size,random_state=CFG.seed,stratify=Y)
    schema = infer_basic_schema(X_train)

    print("\nPhase 1 — Data & Split Design")
    print("Dataset:", spec.filename, "| target:", spec.target_col)
    print("\nShapes")
    print("X_train:", X_train.shape, "y_train:", Y_train.shape)
    print("X_test :", X_test.shape, "y_test :", Y_test.shape)

    print("\nTarget Distribution (train)")
    print(Y_train.value_counts(normalize=True).round(3))

    print("\nTarget Distribution (test)")
    print(Y_test.value_counts(normalize=True).round(3))

    print("\nSchema")
    print("Numeric cols     :", schema.numeric)
    print("Categorical cols :", schema.categorical)

    print("\nSanity Checks")
    print("- Cached file exists:", (titdf is not None))

if __name__ =='__main__':
    main()