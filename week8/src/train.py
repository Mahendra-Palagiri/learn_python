from data_load import DatasetSpec,load_dataset
import pandas as pd

spec = DatasetSpec(
    source="seaborn",
    filename="titanic_seaborn_cached.csv",
    target_col="survived",
    seaborn_name="titanic",
)

titdf = load_dataset(spec)
print(titdf.info())
print(titdf.describe(include='all'))