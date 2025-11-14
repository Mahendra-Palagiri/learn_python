import numpy as np
import pandas as pd
from pathlib import Path

def generate_titanic_like(n_rows=10000, random_state=42):
    rng = np.random.default_rng(random_state)

    # --- Basic distributions ---
    # Age ~ normal(30, 14), clipped to [0.4, 80]
    age = rng.normal(loc=30, scale=14, size=n_rows)
    age = np.clip(age, 0.4, 80)

    # Fare ~ lognormal to get skew (most low fares, few very high)
    fare = rng.lognormal(mean=3.0, sigma=0.7, size=n_rows)  # ~20–100+ typical
    fare = np.clip(fare, 5, 300)

    # Sex ~ 0/1 then map to strings
    sex_raw = rng.integers(0, 2, size=n_rows)
    sex = np.where(sex_raw == 0, "male", "female")

    # Embarked: S most common, then C, then Q
    embarked = rng.choice(["S", "C", "Q"], size=n_rows, p=[0.6, 0.25, 0.15])

    # Cabin: some random letters + numbers, but many missing
    cabin_letters = rng.choice(list("ABCDEF"), size=n_rows)
    cabin_numbers = rng.integers(1, 250, size=n_rows)
    cabin = np.array([f"{l}{n}" for l, n in zip(cabin_letters, cabin_numbers)])

    # Introduce missing values in Age, Fare, Cabin (~10–15%)
    mask_age_nan = rng.random(n_rows) < 0.12
    mask_fare_nan = rng.random(n_rows) < 0.08
    mask_cabin_nan = rng.random(n_rows) < 0.7  # most cabins missing

    age = age.astype("float")
    fare = fare.astype("float")

    age[mask_age_nan] = np.nan
    fare[mask_fare_nan] = np.nan
    cabin[mask_cabin_nan] = ""

    # --- Define a "true" survival model (this is the important part) ---
    # We'll build a logit:
    # logit(p) = b0 + b_age * Age + b_fare * Fare + b_female * I(female) + b_embC * I(Embarked=C)

    # Encode for our internal formula
    is_female = (sex == "female").astype(float)
    embarked_C = (embarked == "C").astype(float)

    # Coefficients chosen to give intuitive behavior:
    b0 = -2.0           # base survival is low
    b_age = -0.02       # older age → slightly lower survival
    b_fare = 0.01       # higher fare → higher survival
    b_female = 1.5      # females have much higher survival odds
    b_embC = 0.5        # C embarkation slightly better

    # For missing Age/Fare, temporarily fill with reasonable defaults for generating p
    age_for_model = np.where(np.isnan(age), 30, age)
    fare_for_model = np.where(np.isnan(fare), 30, fare)

    logit_p = (
        b0
        + b_age * age_for_model
        + b_fare * (fare_for_model / 10.0)  # scale fare effect down a bit
        + b_female * is_female
        + b_embC * embarked_C
    )

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    p_survive = sigmoid(logit_p)

    # Sample Survived from Bernoulli(p)
    survived = rng.binomial(1, p_survive, size=n_rows)

    df = pd.DataFrame(
        {
            "Age": age,
            "Fare": fare,
            "Embarked": embarked,
            "Cabin": cabin,
            "Survived": survived,
            "Sex": sex,
        }
    )

    return df


if __name__ == "__main__":
    # Adjust path as needed
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "titanic_synthetic.csv"

    df = generate_titanic_like(n_rows=10000, random_state=42)
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic dataset to: {out_path}")
    print(df.head())
    print(df['Survived'].value_counts(normalize=True))