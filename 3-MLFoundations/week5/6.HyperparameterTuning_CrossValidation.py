''' Week 5 · Day 6 — Hyperparameter Tuning & Cross-Validation

🎯 Learning Goal

By the end of today, we will understand:
	•	Why train/test accuracy is not enough
	•	What cross-validation really measures
	•	Why GridSearchCV is not just “trying combinations”
	•	How to tune C, penalty, solver properly
	•	How to avoid accidentally overfitting the test set

This is where our work starts to resemble real ML practice.

'''

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report


titdf = pd.read_csv('./data/week5/titanic_synthetic.csv')

num_cols = ['Age','Fare']
cat_cols = ['Embarked','Sex']
features = num_cols+cat_cols
target = 'Survived'

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scaler',RobustScaler())
])

cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

pre_prcsr = ColumnTransformer([
    ('nums',num_pipe,num_cols),
    ('catg',cat_pipe,cat_cols)
])

final_pipe = Pipeline([
    ('preprcsr',pre_prcsr),
    ('model',LogisticRegression(max_iter=1000))
])

X = titdf[features]
Y = titdf[target]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

param_grid = {
    'model__C':[0.01,0.1,1,10],
    'model__penalty':['l1','l2'],
    'model__solver': ['liblinear']
}

grid = GridSearchCV(
    estimator=final_pipe,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid.fit(X_train,Y_train)

# print('\n\n Best Params', grid.best_params_)
# print('\n\n Best Score', grid.best_score_)

best_estimator = grid.best_estimator_
bestYpred = best_estimator.predict(X_test)
# print(classification_report(Y_test,bestYpred))

cv_df = pd.DataFrame(grid.cv_results_)
cv_df[['params', 'mean_test_score', 'std_test_score']].sort_values(
    by='mean_test_score', ascending=False
).head()
print(cv_df)





''' ------------ ** Theory ----------------------
1️⃣ Why tuning is necessary (even when metrics look fine)

From Day 5 you saw:
	•	Changing C or penalty didn’t change accuracy much
	•	Everything looked “stable”

That’s exactly the danger zone.

Why?

Because:
	•	One train/test split can be lucky
	•	Another split may behave differently
	•	You don’t know if your chosen C=1.0 is robust

So we ask a better question:

Which hyperparameters perform well across many data splits?

That’s cross-validation.

⸻

2️⃣ What cross-validation really is (no buzzwords)

Instead of:
	•	Train once
	•	Test once

We do:
	•	Split data into K folds
	•	Train on K-1 folds
	•	Validate on the remaining fold
	•	Repeat K times
	•	Average the results

This answers:

“How stable is this model across different samples of data?”

⸻

3️⃣ Why we tune hyperparameters with CV (not the test set)

Very important rule:

The test set is sacred.
You do NOT tune on it.

So:
	•	Training set → used for CV
	•	Validation (inside CV) → used to pick hyperparameters
	•	Test set → used once at the end

GridSearchCV enforces this discipline.

⸻

4️⃣ GridSearchCV — what it really does

GridSearchCV:
	•	Takes a pipeline
	•	Takes a parameter grid
	•	Performs cross-validation for every combination
	•	Picks the one with the best mean CV score

It is NOT:
	•	“Try everything and pick the highest test accuracy”

It IS:
	•	“Find the most stable configuration”

⸻

5️⃣ Define the parameter grid (logistic regression–specific)

We’ll tune:
	•	C
	•	penalty
	•	solver
'''