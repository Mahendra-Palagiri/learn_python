'''Week 7 — Day 5: Hyperparameter Tuning with CV (GridSearch vs RandomizedSearch)

📚 Topics
	•	Hyperparameters vs parameters (what gets tuned vs learned)
	•	GridSearchCV vs RandomizedSearchCV (when to use which)
	•	Tuning on the Pipeline (still leakage-proof)
	•	Reading results: best params, CV mean, CV std, train vs test CV gap
	•	(Bonus) Treating StandardScaler vs RobustScaler as a tunable choice

🎯 Learning Goals
	•	Run a CV-based hyperparameter search on the same Day 4 pipeline
	•	Improve (or at least benchmark) F1 mean vs ~0.65 while keeping std reasonable
	•	Produce evidence: top configs + stability (std)

'''

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV,RandomizedSearchCV

#1. Import the titanic data set
titdf = pd.read_csv('./data/week5/titanic_synthetic.csv')
# print(titdf.info())
# print(titdf.describe(include='all'))

#2. Categorize Columns and split the dataframe
num_cols =['Age','Fare']
cat_cols=['Sex','Embarked']
target ='Survived'
features = num_cols +cat_cols
X = titdf[features]
Y = titdf[target]

#3. Create basic pipelines
num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='mean')),
    ('scale',StandardScaler())
])

cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(handle_unknown='ignore'))
])

#4. Create Pre-Processor and Final Pipleline
pre_prcsr = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])

final_pipe = Pipeline([
    ('preprcsr',pre_prcsr),
    ('model', LogisticRegression(max_iter=1000,solver='liblinear'))
])

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

param_grid = {
    # model knobs
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l1", "l2"],
    "model__class_weight": [None, "balanced"],

    # scaler knob (optional but recommended)
    "preprcsr__num__scale": [StandardScaler(), RobustScaler()]
}

gscv = GridSearchCV(
    estimator=final_pipe,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

gscv.fit(X,Y)

print("Best params:", gscv.best_params_)
print("Best CV F1:", gscv.best_score_)


results = pd.DataFrame(gscv.cv_results_).sort_values("mean_test_score", ascending=False)
top3 = results.head(3)[["mean_test_score", "std_test_score", "mean_train_score", "params"]]
print("\nTop 3 configs:\n", top3.to_string(index=False))

'''
Best params: {'model__C': 0.01, 'model__class_weight': None, 'model__penalty': 'l1', 'preprcsr__num__scale': StandardScaler()}
Best CV F1: 0.6495737145596958

Top 3 configs:
  mean_test_score  std_test_score  mean_train_score                                                                                                                  params
        0.649574        0.013113          0.649581       {'model__C': 0.01, 'model__class_weight': None, 'model__penalty': 'l1', 'preprcsr__num__scale': StandardScaler()}
        0.649574        0.013113          0.649581 {'model__C': 0.01, 'model__class_weight': 'balanced', 'model__penalty': 'l1', 'preprcsr__num__scale': StandardScaler()}
        0.649574        0.013113          0.649581   {'model__C': 0.01, 'model__class_weight': 'balanced', 'model__penalty': 'l1', 'preprcsr__num__scale': RobustScaler()}

'''


#=========================================================
# RetroSpection
#=========================================================
'''
That param_grid is the menu of settings GridSearchCV tries. 
It trains the pipeline many times, once per combination, and picks the best by CV F1.

How to read the keys like "model__C"

In a Pipeline, each step has a name (like "preprcsr" and "model").
stepname__parametername means: “set this parameter on that step.”

So:
	•	model__C = parameter C on the "model" step (LogisticRegression)
	•	preprcsr__num__scale = parameter scale on the numeric sub-pipeline inside "preprcsr" 
                                (the ColumnTransformer branch "num")

⸻

What each grid item means

1) "model__C": [0.01, 0.1, 1, 10, 100]

This is LogisticRegression’s regularization strength control.
	•	C is inverse of regularization strength.
	•	Small C (0.01) → strong regularization → simpler model → less overfitting, more bias risk
	•	Large C (100) → weak regularization → more flexible model → less bias, more overfitting risk

So the grid is testing: “Do we do better with a tighter, simpler model or a looser, more flexible one?”

⸻

2) "model__penalty": ["l1", "l2"]

This chooses the type of regularization.
	•	L2 (ridge-style): shrinks all coefficients smoothly toward 0 (rarely exactly 0).
        Good default, stable when features correlate.
	•	L1 (lasso-style): can push some coefficients to exactly 0 (acts like feature selection).
        Useful when many features are irrelevant, but can be more “picky”/unstable.

You used solver="liblinear" which supports both l1 and l2 for LogisticRegression—so this grid is valid.

⸻

3) "model__class_weight": [None, "balanced"]

This controls whether the model should treat classes equally or compensate for imbalance.
	•	None → treat each row equally (default)
	•	"balanced" → automatically up-weights the minority class and down-weights 
                        the majority class (based on class frequencies)

Why it matters:
	•	If positive class is rare, a model can look decent but miss positives. 
            "balanced" often improves recall/F1.

⸻

4) "preprcsr__num__scale": [StandardScaler(), RobustScaler()]

This is treating the scaler choice as a hyperparameter.
	•	StandardScaler: subtract mean, divide by standard deviation. 
                        Works well when distributions are roughly normal-ish and outliers aren’t extreme.
	•	RobustScaler: subtract median, divide by IQR. More resistant to outliers.

Why tune it?
	•	Outliers (like a long-tailed Fare) can distort StandardScaler and hurt linear models. RobustScaler can help.
	•	But sometimes RobustScaler doesn’t help (as we saw—both were tied).

Important: this only affects the numeric columns branch ("num"). Categorical columns go through OneHotEncoder and aren’t scaled.

⸻

What GridSearchCV actually tries

Total combinations here:
	•	C (5) × penalty (2) × class_weight (2) × scaler (2) = 40 configs
With 5-fold CV → 40 × 5 = 200 fits.

That’s why GridSearch can get expensive quickly.



'''
