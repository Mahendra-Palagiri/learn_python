''' Week 6 · Day 7 — Feature Engineering + Non-Linear Baseline (so we actually move the needle)

Topics
	•	Why our linear model hits a ceiling (cap at 5 + non-linear relationships)
	•	Feature engineering options:
        •	simple transforms (log / sqrt)
        •	interaction terms
	•	A non-linear baseline model (tree-based) for comparison:
        •	DecisionTreeRegressor
        •	RandomForestRegressor (strong baseline)
	•	Compare to our LinearRegression baseline using the same train/val/test discipline

🎯 Learning Goals

By the end of Day 7, we will be able to:
	•	Build a non-linear regression baseline and compare it fairly to linear regression
	•	Use validation to choose a simple hyperparameter (tree depth / number of trees)
	•	Diagnose overfitting by comparing train vs validation performance
	•	Decide what direction Week 6 should take next (linear improvements vs non-linear models)
'''

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# ==========================================================
# Step 1 — Helper method to look at the metrics
# ==========================================================
def regression_report(y_true,y_pred, lbl: str):
    mae = mean_absolute_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    r2 = r2_score(y_true,y_pred)

    print(f"\n======================== Regression Report for ::  '{lbl}' =======================")
    print(f"Mean Absolute Error : {mae}")
    print(f"Root Mean Squared Error : {rmse}")
    print(f"R2 Score : {r2}")

    return mae,rmse,r2


# ==========================================================
# Step 2 — Load and split data
# ==========================================================
chdf = fetch_california_housing(as_frame=True).frame
target ='MedHouseVal'

X = chdf.drop(columns=[target])
Y = chdf[target]

X_trainval,X_test,Y_trainval,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
X_train,X_val,Y_train,Y_val = train_test_split(X_trainval,Y_trainval,test_size=0.2,random_state=42)


# ==========================================================
# Step 3 — Baseline model 
# ==========================================================
lr = LinearRegression()
lr.fit(X_train,Y_train)

lr_ypred_val = lr.predict(X_val)
lr_mae,lr_rmse,lr_r2 = regression_report(Y_val,lr_ypred_val, "Linear Regression (Validation)")


# ==========================================================
# Step 4 — Decision Tree regression Model
# ==========================================================

print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

depths = [2,5,7,10,12, None]

dt_train_rmse = None
dt_val_rmse = None
dt_optimal_depth=None

for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth,random_state=42)
    dt.fit(X_train,Y_train)

    dt_ypred_train = dt.predict(X_train)
    dt_ypred_val = dt.predict(X_val)
    _,curr_train_rsme,_ = regression_report(Y_train,dt_ypred_train,f"Decision Tree Regression model with depth {depth} (Train)")
    _,curr_val_rsme,_ = regression_report(Y_val,dt_ypred_val,f"Decision Tree Regression model with depth {depth} (Validation)")

    if dt_optimal_depth is None:
        dt_optimal_depth=depth

    if dt_train_rmse is None:
        dt_train_rmse = curr_train_rsme

    if dt_val_rmse is None:
        dt_val_rmse = curr_val_rsme
        continue

    if curr_train_rsme < dt_train_rmse and curr_val_rsme < dt_val_rmse:
        print("\n~~~~~ Training Model Improvement~~~~~")
        dt_train_rmse = curr_train_rsme
        dt_val_rmse = curr_val_rsme
        dt_optimal_depth = depth
    else:
        print("\n Train validation RSME has detoriated")
        break

print(f"\nFinal Values  Optimal Depth ::  {dt_optimal_depth} \t Train_RMSE :: {dt_train_rmse} \t Val RMSE :: {dt_val_rmse}")

print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

'''
What we watch for
	•	If train RMSE becomes tiny but val RMSE gets worse → overfitting.
    •	Pick the depth that gives the best validation RMSE.
'''


# ==========================================================
# Step 5 — Random Forest
# ==========================================================
# Random forest reduces overfitting vs a single tree
candidates = [
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 200, "max_depth": 12},
    {"n_estimators": 400, "max_depth": 12},
]

best = None
best_model = None

for c in candidates:
    rf = RandomForestRegressor(
        n_estimators=c["n_estimators"],
        max_depth=c["max_depth"],
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, Y_train)

    pred_val = rf.predict(X_val)
    mae, rmse, r2 = regression_report(Y_val, pred_val, f"RandomForest (Val) {c}")

    if best is None or rmse < best:
        best = rmse
        best_model = rf

print("\nBest RandomForest validation RMSE:", best)

pred_test_rf = best_model.predict(X_test)
_,rand_test_rsme,_=regression_report(Y_test, pred_test_rf, "Best RandomForest (Test)")

print(f"\n\n\n\n  -- Final rsme Values  \n LR : {lr_rmse} \n DT:  {dt_val_rmse} \n RF: {rand_test_rsme} ")

'''
-- Final rsme Values  
 LR : 0.7338977899061466 
 DT:  0.6651367086127731 
 RF: 0.5134205320863233 
'''

# ==========================================================
# RETROSPECTION
# ==========================================================

'''
Q) Explain a bit more about random forest

    What a Random Forest is (simple mental model)

        A Random Forest regressor is:

        a bunch of decision trees, each trained a little differently, whose predictions we average.

        One tree is like one opinion. A forest is like asking 300 reasonably smart people and averaging their guesses.

        That averaging is the superpower: it reduces the “wild swings” a single tree makes.

    ⸻

    Step 1: What a single Decision Tree is doing

        A decision tree learns rules like:
            •	if MedInc > 3.5 go right, else left
            •	then if Latitude < 34 go right, else left
            •	keep splitting until we reach a leaf
            •	in each leaf, predict the average y of training points that landed there

        So a tree creates a bunch of little “regions” in feature space and predicts a constant value per region.

        Why a single tree overfits

        If we allow it to grow deep, it can make tiny regions that match training data too perfectly (memorization). That’s why DT often gets low training error but worse validation error.

    ⸻

    Step 2: Why we build many trees

        If we train many deep trees, they’ll overfit in different ways.

        Then when we average them:
            •	the random mistakes cancel out
            •	the signal remains

        That’s the “ensemble” idea.

    ⸻

    Step 3: Where the “random” comes from (2 sources of randomness)

        Randomness #1: Bootstrapping rows (bagging)

        Each tree is trained on a random sample of the training rows, with replacement.
        That sample is called a “bootstrap sample.”

        So tree A sees a slightly different dataset than tree B.

        This reduces variance (stabilizes predictions).

        Randomness #2: Random subset of features at each split

        At every split, instead of considering all features, the tree considers only a random subset (e.g., maybe 3 out of 8 features).

        That forces trees to diversify:
            •	one tree might lean heavily on MedInc
            •	another might discover a strong geo split (Latitude/Longitude)
            •	another might use occupancy patterns

        This prevents every tree from becoming a clone.

    ⸻

    Step 4: How a Random Forest predicts (regression)

        For a new data point x:
            1.	send x down tree 1 → prediction p_1
            2.	send x down tree 2 → prediction p_2
        …
        N) send x down tree N → prediction p_N

        Final prediction:
        Yˆ = (p_1 + p_2 + ..... + p_N)/N

        That’s it.

    ⸻

    How this maps to the code we ran

        A typical call looked like:
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        pred_val = rf.predict(X_val)

        Key parameters
            •	n_estimators: number of trees
        More trees → more stable (but slower). After a point, gains flatten.
            •	max_depth: how deep each tree can grow
        Deeper → more complex, can overfit more
        Shallower → simpler, may underfit
            •	random_state: makes randomness repeatable so our results are consistent
            •	n_jobs=-1: use all CPU cores

    ⸻

    Why our Random Forest RMSE got much better than Linear Regression

        Linear regression can only learn a plane:
        yˆ = b + w_1x_1 + ........

        Random Forest can learn patterns like:
            •	“income matters a lot, but only in certain regions”
            •	“room count matters differently for high income vs low income”
            •	“latitude and longitude create distinct price bands”
        These are interactions and non-linear effects.

        That’s why we saw:
            •	LR RMSE ~0.734
            •	RF RMSE ~0.513

        A forest can model more of the real structure.

'''