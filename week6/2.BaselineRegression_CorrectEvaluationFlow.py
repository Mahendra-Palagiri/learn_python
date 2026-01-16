''' Week 6 · Day 2 — Baseline Regression + Correct Evaluation Flow

Topics
	•	What a baseline is and why we always start with it
	•	Proper train / validation / test workflow (no leakage)
	•	Regression metrics: MAE, MSE, RMSE, R^2
	•	Visual checks: pred vs actual, residuals
	•	Keeping the test set sacred (only once at the end)

🎯 Learning Goals

By the end of today, we will be able to:
	•	Build a clean regression workflow with train/val/test splits
	•	Create a baseline predictor and verify our model beats it
	•	Train a LinearRegression model in scikit-learn the right way
	•	Compute and interpret: MAE, RMSE, R^2
	•	Use plots to sanity-check behavior and detect obvious issues
	•	Explain what “good” looks like (and what’s suspicious)
'''

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# 1️⃣ ~~ Fetch Dataset 
chdf = fetch_california_housing(as_frame=True).frame

# print(chdf.info())
# print(chdf.describe(include='all'))
# print(chdf.shape)
# print(chdf.columns)

# 2️⃣ ~~ Prepare for train, validate and test
target='MedHouseVal'
X = chdf.drop(columns=[target])
Y = chdf[target]

# 80% of data for training and validation (remaining 20% for testing)
X_trainval,X_test,Y_trainval,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# Among the 80% of data previously selected for training and valdiation (split that portion 80% for training and 20% for validation)
X_train,X_val,Y_train,Y_val = train_test_split(X_trainval,Y_trainval,test_size=0.2,random_state=42)

# print("\n Split Sizes:")
# print("Original data Set : \t", X.shape, Y.shape)
# print("Training Set : \t", X_train.shape, Y_train.shape)
# print("Validation Set : \t", X_val.shape, Y_val.shape)
# print("Testing Set : \t", X_test.shape, Y_test.shape)


# 3️⃣ ~~~ Base Model ~~~

# Creating an array to the length of Y_Val where each value is --> mean of the training set
y_pred_baseline_val = np.full(len(Y_val),fill_value=Y_train.mean())

mae_base = mean_absolute_error(Y_val,y_pred_baseline_val)
rmse_base = np.sqrt(mean_squared_error(Y_val,y_pred_baseline_val)) # Root Mean Square Error
r2_base = r2_score(Y_val,y_pred_baseline_val)

print("\n ~~~~~~~ Baseline validation ~~~~~~~~")
print("MAE \t:",mae_base)
print("RMSE \t:",rmse_base)
print("R2 \t:",r2_base)

'''
Interpretation:
	•	R² baseline can be ~0 or negative. That’s fine.
	•	wer model must beat this baseline to be worth anything
'''

# 4️⃣ ~~~~ Introducing LinearRegression ~~~~

lr = LinearRegression()
lr.fit(X_train,Y_train)

y_pred_lr_val = lr.predict(X_val)

mae_lr = mean_absolute_error(Y_val,y_pred_lr_val)
rmse_lr = np.sqrt(mean_squared_error(Y_val,y_pred_lr_val))
r2_lr = r2_score(Y_val,y_pred_lr_val)

print("\n ~~~~~~~~~~~~~LINEAR REGRESSION (Val) ~~~~~~~~~~~")
print("MAE :", mae_lr)
print("RMSE:", rmse_lr)
print("R2  :", r2_lr)

'''
Expected: MAE/RMSE should improve; R² should increase vs baseline.
'''

# 5️⃣ ~~~ Visual Check ~~~
plt.figure()
plt.scatter(Y_val,y_pred_lr_val)
plt.xlabel("Actual Y (Val)")
plt.ylabel("Predicted Y(Val)")
plt.title("Predicted vs Actual (Validation)")
# plt.show()

residuals = y_pred_lr_val-Y_val
plt.figure()
plt.scatter(y_pred_lr_val,residuals)
plt.axhline(0)
plt.xlabel("Predicted Y (Val)")
plt.ylabel("Residual (Pred-Actual)")
plt.title("Residuals vs predicted (Validation)")
# plt.show()

# 6️⃣ ~~~ Evaluation on test data ~~~
y_lr_test = lr.predict(X_test)
mae_lr_test = mean_absolute_error(Y_test,y_lr_test)
rmse_lr_test = np.sqrt(mean_squared_error(Y_test,y_lr_test))
r2_lr_test = r2_score(Y_test,y_lr_test)

print("\n ~~~~~~~ LINEAR REGRESSION (test) ~~~~~~`")
print("MAE :", mae_lr_test)
print("RMSE:", rmse_lr_test)
print("R2  :", r2_lr_test)


''' ~~~~~~~ Retrospection ~~~~~~~

Q) what we are doing in step 3 and why (Baseline model) why are we constructing something outside of our original dataset we selected)

        1) We are NOT creating a new dataset

        We’re using the same dataset we selected.

        From that dataset we split into:
            •	train
            •	validation
            •	test

        The baseline uses only:
            •	y_train (the true target values in the training split)

        So it’s 100% derived from wer original dataset.

        ⸻

        2) What is the baseline model?

        A baseline model is the dumbest reasonable predictor.

        For regression, the most common baseline is:

        Predict the average of the training targets for every case.

        So if the average house value in y_train is 2.06 (in the dataset’s units), then the baseline predicts:
            •	2.06 for every row in validation

        That’s it.

        No features, no learning relationships — just one constant guess.

        ⸻

        3) Why do we need a baseline?

        Because otherwise we can fool ourself.

        A “real model” can look impressive, but might be:
            •	barely better than guessing
            •	or worse than guessing
            •	or only good because of leakage

        Baseline gives we a sanity bar:

        “If our fancy model can’t beat this dumb guess, it’s not useful.”

        This is the regression equivalent of what we did in Week 5:
            •	In classification, a common baseline is “always predict the majority class.”
            •	If our classifier can’t beat that, it’s worthless.

        Same exact principle.

        ⸻

        4) Why is the baseline “mean of y_train” specifically?

        Because it is the constant prediction that minimizes squared error among all constant predictions.

        In plain English:
            •	If we are forced to predict the same number for everyone,
            •	the best single number (for MSE) is the average.

        So it’s not random—it’s the best “no-features” strategy.

        ⸻

        5) What does baseline tell we?

        It tells us how hard the problem is.
            •	If baseline RMSE is already low, the target might be easy or low-variance.
            •	If baseline is bad, there’s room for a model to improve.

        And it gives we a simple sentence for interviews / real work:

        “Linear regression improved RMSE by X% compared to the baseline mean predictor.”

Q) whey do we create this array for baseline. --> y_pred_baseline_val = np.full(shape=len(y_val), fill_value=y_train.mean())

        What this produces

        It creates an array of predictions for the validation set, where every prediction is the same number.

        That number is:
            •	y_train.mean() = the average target value in our training data

        So if the average of y_train is, say, 2.05, and our validation set has 3,000 rows, then:

        y_pred_baseline_val becomes:

        [2.05, 2.05, 2.05, ..., 2.05] (3,000 times)

        Why we do this

        We are building a baseline predictor: “If I refuse to use any features (X) at all, what’s the best dumb guess I can make?”
            •	For MSE/RMSE, the best single constant guess is the mean.
            •	For MAE, the best single constant guess is the median.

        So we use the mean baseline to say:

        “Here’s how good we can do with zero intelligence.”

        Then we check if Linear Regression does better.

        Why use len(y_val) for the shape?

        Because we want one predicted value per validation row, so we can compute metrics on validation:
            •	mean_absolute_error(y_val, y_pred_baseline_val)
            •	rmse, r2, etc.

        If validation has N rows, predictions must also have N values.

        Why is this NOT “outside the dataset”?

        Because the baseline value (y_train.mean()) comes directly from our dataset:
            •	training split only (to avoid peeking)

        We avoid using validation/test to set the baseline because that would leak information.
'''