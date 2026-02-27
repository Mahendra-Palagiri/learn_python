'''Week 6 · Day 6 — Influence removal vs Ridge regularization

Topics
    • Remove top-K influential rows (training only) and compare
    • Coefficient stability comparison
    • Ridge regression (regularization) as a safer alternative

🎯 Learning Goals
    • Run a disciplined influence-removal experiment (no leakage)
    • Understand when removing points helps vs hurts
    • Use Ridge to stabilize coefficients under multicollinearity
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================================
# Step 1 — Helper method to look at the metrics
# ==========================================================
def regression_report(y_true, y_pred, label: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n======================================{label}========================================")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)
    return mae, rmse, r2


# ==========================================================
# Step 2 — Load and split data
# ==========================================================
chdf = fetch_california_housing(as_frame=True).frame

target='MedHouseVal'
X= chdf.drop(columns=[target])
Y= chdf[target]

X_trainval,X_test,Y_trainval,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
X_train,X_val,Y_train,Y_val = train_test_split(X_trainval,Y_trainval,test_size=0.2,random_state=42)



# ==========================================================
# Step 3  - Baseline real model on 'full' training set
# ==========================================================
lr_full = LinearRegression()
lr_full.fit(X_train,Y_train)

lr_full_y_pred = lr_full.predict(X_val)
mae_full,rmse_full,r2_full= regression_report(Y_val,lr_full_y_pred,"LinearRegression (Validation) - Full training")

coef_full = pd.Series(lr_full.coef_,index=X_train.columns,name="coef_full")

'''
======================================LinearRegression (Validation) - Full training========================================
MAE : 0.5407140746076876
RMSE: 0.7338977899061466
R2  : 0.6097023914123473
'''


# ==========================================================
# Step 4  - Compute influence on training set (OLS Cook D)
# ==========================================================
X_train_sm = sm.add_constant(X_train)
ols = sm.OLS(Y_train,X_train_sm).fit()

influence = ols.get_influence()
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag
stud_resid = influence.resid_studentized_internal

top_k = 10
top_idx = np.argsort(cooks_d)[-top_k:][::-1]

'''
print("\n==============================================================================")
print("Top influential training points by Cook's D:")
for rank, idx in enumerate(top_idx, 1):
    print(f"{rank:2d}) idx={idx}  CookD={cooks_d[idx]:.6f}  leverage={leverage[idx]:.6f}  stud_resid={stud_resid[idx]:.3f}")


Top influential training points by Cook's D:
 1) idx=2757  CookD=2.001294  leverage=0.576413  stud_resid=3.638
 2) idx=2580  CookD=1.904147  leverage=0.403434  stud_resid=5.034
 3) idx=225  CookD=0.484712  leverage=0.117223  stud_resid=-5.732
 4) idx=10876  CookD=0.135771  leverage=0.067223  stud_resid=4.118
 5) idx=3798  CookD=0.123410  leverage=0.054898  stud_resid=-4.373
 6) idx=9659  CookD=0.051227  leverage=0.051593  stud_resid=2.911
 7) idx=1080  CookD=0.037541  leverage=0.004830  stud_resid=-8.343
 8) idx=4023  CookD=0.036925  leverage=0.033731  stud_resid=-3.085
 9) idx=10332  CookD=0.036688  leverage=0.033101  stud_resid=-3.106
10) idx=2308  CookD=0.026171  leverage=0.047587  stud_resid=-2.171
'''

# ==========================================================
# Step 5 — Remove top-K influential rows (TRAINING ONLY) and refit
# ==========================================================
# We remove rows only from X_train / y_train, 
# never from validation/test:

mask_keep = np.ones(len(X_train), dtype=bool)
mask_keep[top_idx] = False  # drop top influencers

X_train_clean = X_train.iloc[mask_keep]
Y_train_clean = Y_train.iloc[mask_keep]

'''
    print("\nTraining rows before:", len(X_train))
    print("Training rows after :", len(X_train_clean))

    Training rows before: 13209
    Training rows after : 13199
'''

#Refit
lr_clean = LinearRegression()
lr_clean.fit(X_train_clean,Y_train_clean)

lr_clean_y_pred = lr_clean.predict(X_val)
mae_clean,rmse_clean,r2_clean= regression_report(Y_val,lr_clean_y_pred,"LinearRegression (Validation) - Cleaned set (dropped top 10 rows)")

'''
======================================LinearRegression (Validation) - Cleaned set (dropped top 10 rows)========================================
MAE : 0.5854153979414096
RMSE: 3.5682323782109675
R2  : -8.22637261541536
'''


coef_clean = pd.Series(lr_clean.coef_,index=X_train.columns,name="coef_clean")

coef_compare = pd.concat([coef_full, coef_clean], axis=1)
coef_compare["abs_change"] = (coef_compare["coef_clean"] - coef_compare["coef_full"]).abs()
coef_compare["pct_change"] = coef_compare["abs_change"] / (coef_compare["coef_full"].abs() + 1e-12) * 100

'''
print("\n==============================================================================")
print("Top 10 coefficient changes after dropping influential points:")
print(coef_compare.sort_values("abs_change", ascending=False).head(10))

Top 10 coefficient changes after dropping influential points:
            coef_full  coef_clean  abs_change   pct_change
AveOccup    -0.003971   -0.162477    0.158506  3991.726960
AveBedrms    0.781543    0.840408    0.058865     7.531878
Longitude   -0.429343   -0.420949    0.008395     1.955251
AveRooms    -0.125858   -0.130162    0.004304     3.420026
HouseAge     0.009612    0.010466    0.000854     8.884039
Latitude    -0.416855   -0.417131    0.000276     0.066315
MedInc       0.451677    0.451837    0.000159     0.035271
Population  -0.000007    0.000015    0.000022   303.883765

What we learn here
	•	If metrics barely change but coefficients swing a lot → those points were stabilizing a relationship or we have multicollinearity.
	•	If metrics improve a lot and coefficients stabilize → those points were distorting the model.
	•	If metrics get worse → those points were legitimate and important.
'''

# ==========================================================
# Step 6 — Ridge (regularization) as a safer stabilizer
# ==========================================================
'''
    Ridge shrinks coefficients, usually helping when features are correlated.

    We use a pipeline because Ridge is sensitive to scale:
'''

alphas = [0.01, 0.1, 1, 10, 100]

best = None
best_alpha = None

for a in alphas:
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=a, random_state=42))
    ])
    ridge.fit(X_train, Y_train)
    pred_val = ridge.predict(X_val)
    mae, rmse, r2 = regression_report(Y_val, pred_val, f"Ridge (Validation) alpha={a}")
    if best is None or rmse < best:
        best = rmse
        best_alpha = a

print("\nBest alpha by validation RMSE:", best_alpha)

# Train final Ridge with best alpha and do one-time test evaluation:
ridge_best = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=best_alpha, random_state=42))
])
ridge_best.fit(X_train, Y_train)

pred_test = ridge_best.predict(X_test)
regression_report(Y_test, pred_test, f"Ridge (Test) alpha={best_alpha}")

'''
======================================Best Ridge (Test) alpha=0.01========================================
MAE : 0.5334737322292816
RMSE: 0.7446288583585975
R2  : 0.5768709266287578
'''

# ==========================================================
# RETROSPECTION
# ==========================================================

'''
Understanding outputs:
    ======================================LinearRegression (Validation) - Full training========================================
    MAE : 0.5407140746076876
    RMSE: 0.7338977899061466
    R2  : 0.6097023914123473

    ======================================LinearRegression (Validation) - Cleaned set (dropped top 10 rows)========================================
    MAE : 0.5854153979414096
    RMSE: 3.5682323782109675
    R2  : -8.22637261541536

    ======================================Ridge (Validation) alpha=0.01========================================
    MAE : 0.5407140609310417
    RMSE: 0.7338978362704273
    R2  : 0.6097023420979368

    ======================================Ridge (Validation) alpha=0.1========================================
    MAE : 0.5407139379458213
    RMSE: 0.7338982543626572
    R2  : 0.6097018974026222

    ======================================Ridge (Validation) alpha=1========================================
    MAE : 0.5407127184263566
    RMSE: 0.7339025156537036
    R2  : 0.6096973649524395

    ======================================Ridge (Validation) alpha=10========================================
    MAE : 0.5407063350323648
    RMSE: 0.7339529824101331
    R2  : 0.6096436848386952

    ======================================Ridge (Validation) alpha=100========================================
    MAE : 0.5412729060492447
    RMSE: 0.7350898788615211
    R2  : 0.6084334208568799

    ======================================Ridge (Test) alpha=0.01========================================
    MAE : 0.5334737322292816
    RMSE: 0.7446288583585975
    R2  : 0.5768709266287578

    1) LinearRegression (Validation) — full training
	•	MAE: 0.54071407
	•	RMSE: 0.73389779
	•	R²: 0.60970239

    This is our baseline “real model” reference.

    2) LinearRegression (Validation) — dropped top 10 Cook’s D rows
        •	MAE: 0.58541540
        •	RMSE: 3.56823238
        •	R²: -8.22637262

    This is a failure case: dropping influential points destroyed generalization.

        What happened when we dropped the top 10 Cook’s D rows?

        It broke the model on validation. Big time.
            •	Full training (Validation): RMSE 0.734, R² 0.610
            •	Dropped top 10 (Validation): RMSE 3.568, R² -8.226

        What that means

        If R² is very negative, our model is performing worse than the dumb baseline on validation. That’s a sign we removed points that were not “bad noise” — they were structurally important for the linear fit (or our removal created numerical instability).

        So: blind removal is not optimization. It’s data amputation.

    3) Ridge (Validation) — sweep

    We pick “best alpha” by lowest validation RMSE.
        •	alpha=0.01 → RMSE 0.73389784
        •	alpha=0.1  → RMSE 0.73389825
        •	alpha=1    → RMSE 0.73390252
        •	alpha=10   → RMSE 0.73395298
        •	alpha=100  → RMSE 0.73508988

    ✅ Best alpha by validation RMSE is 0.01 (barely, but it is the minimum).

    4) Ridge (Test) with best alpha (=0.01)
        •	MAE: 0.53347373
        •	RMSE: 0.74462886
        •	R²: 0.57687093

    That’s our one-time test check for the chosen alpha.

    ⸻

Understanding Coefficients change:

        This is not “optimization output”; 
        it’s a stability check: did our model’s understanding of feature effects change when we removed the most influential rows?

        We computed:
            •	coef_full: coefficient learned using all training rows
            •	coef_clean: coefficient learned after dropping top 10 Cook’s D rows (training only)
            •	abs_change = |coef_clean − coef_full|
            •	pct_change = abs_change / (|coef_full| + tiny) * 100

        So the table is telling us: which feature weights are most sensitive to those influential rows.

        ⸻

        How we interpret the columns (simple)

        coef_full vs coef_clean

        These are the “slopes” for each feature:
            •	a positive coefficient means increasing that feature tends to increase predicted house value (holding others constant)
            •	negative means decreasing predicted value

        abs_change

        How much the slope moved in absolute terms.

        pct_change

        How big that movement is relative to the original coefficient.

        ⚠️ Percent change can look insane when the original coefficient is near zero.

        ⸻

        Now interpret the key rows

        1) AveOccup: -0.003971 → -0.162477 (pct change ~3992%)

        This looks wild, but here’s why:
            •	coef_full is extremely close to 0 (about -0.004)
            •	even a modest absolute move (0.158) becomes a massive percentage

        What it means
            •	Our model’s learned effect of AveOccup is not stable.
            •	Those influential rows were acting like “anchors” for how occupancy relates to house value.
            •	Removing them made the coefficient much more negative, but since the validation performance collapsed (RMSE ~3.57), we should treat coef_clean as coming from a bad model.

        So the real lesson is:

        AveOccup is a feature where the coefficient is highly sensitive to influential points and/or correlated with other features.

        ⸻

        2) Population: -0.000007 → +0.000015 (pct change ~304%)

        This is the classic “tiny coefficient problem” too:
            •	both numbers are extremely close to 0
            •	sign flipped from negative to positive
            •	percent looks big because denominator is tiny

        What it means
            •	Population has a weak and unstable linear effect once we control for other features.
            •	This instability is common with correlated demographic features (Population, Households, AveOccup).

        ⸻

        3) AveBedrms: 0.7815 → 0.8404 (abs change 0.0589, ~7.5%)

        This one is actually meaningful.

        What it means
            •	Even after removing influential rows, the coefficient didn’t explode.
            •	The effect size changed moderately (~0.06), which is not crazy.
            •	But we already know AveBedrms is strongly correlated with AveRooms, so we still interpret cautiously.

        ⸻

        4) Longitude / Latitude: small changes (~2% and ~0.07%)

        These are quite stable:
            •	Longitude: -0.429 → -0.421
            •	Latitude: -0.4169 → -0.4171

        What it means

        The geographic gradient signal is robust; influential rows didn’t change it much.

        ⸻

        5) MedInc: 0.451677 → 0.451837 (almost no change)

        This is very stable.

        What it means

        Median income is the strongest, most reliable linear predictor here. Good.

        ⸻

        The big meta-lesson from this table

        Stable coefficients (good sign)
            •	MedInc, Latitude, Longitude barely changed.
        That suggests their relationships are robust.

        Unstable coefficients (warning sign)
            •	AveOccup, Population changed a lot (especially relative to being near zero).
        This suggests:
            •	multicollinearity / redundancy
            •	sensitivity to leverage points
            •	and that our “cleaned” model is not trustworthy anyway because its validation performance collapsed.

        So we should not interpret coef_clean as “truth”; we interpret it as:

        “Dropping influential points made the model unstable and worse.”

    ⸻
    Explain the diff between the LinearRegression Model and Ridge model

    LinearRegression and Ridge are the same kind of model (a straight line / plane), but they learn the weights differently.

    What both models are trying to do

    Both predict:

    y_pred = b + w_1x_1 + w_2x_2 + ....

    Both want predictions close to actual values.

    ⸻

    LinearRegression (ordinary least squares / OLS)

    Goal: choose w,b to minimize squared error only:

    Loss = sum (y - y_pred)^2

    What this implies
        •	If features are correlated (like AveRooms and AveBedrms), OLS can “spread credit” in unstable ways.
        •	Coefficients can get large or flip signs because multiple coefficient combinations fit almost equally well.
        •	It can be sensitive to high-leverage points (as we saw).

    So OLS is great when:
        •	features are not too correlated,
        •	noise behaves nicely,
        •	we care about the pure best fit under squared error.

    ⸻

    Ridge Regression (L2 regularization)

    Goal: minimize squared error plus a penalty for large coefficients:

    Loss = sum (y - y_pred)^2 + alpha sum (w_j^2)

    That extra term is the key:
        •	it pushes weights to stay smaller unless they really help.

    What this implies
        •	Coefficients become more stable when features are correlated.
        •	Model is less likely to overreact to weird/high-leverage rows.
        •	Often generalizes better (lower validation/test error), especially with multicollinearity.

    Ridge is great when:
        •	predictors are correlated,
        •	we want stability,
        •	we want to reduce variance / overfitting risk.

    ⸻

    What does alpha do?
        •	\alpha = 0 → Ridge becomes basically LinearRegression.
        •	Larger \alpha → more shrinkage → smaller coefficients → sometimes slightly worse fit on training, but better on new data.

    In our run, alpha=0.01 gave almost identical results to LinearRegression — meaning the dataset/model didn’t benefit much from shrinkage at that small value.

    ⸻

    One practical difference we must remember

    Scaling
        •	LinearRegression doesn’t require scaling to work.
        •	Ridge does need scaling (or else features with big units get penalized unfairly).

    ⸻
    What we should conclude (Day 6)
        •	Removing the top influential points is not a good optimization strategy for us here; it made the model much worse on validation.
        •	Ridge with the alphas we tried does not improve validation vs plain LinearRegression; the best alpha is extremely small, meaning regularization isn’t helping much for this specific feature set and model (or we’d need a different alpha range / preprocessing / feature engineering to see gains).
        •	Our model performance is stable: validation RMSE ~0.734 and test RMSE ~0.745 (slightly worse on test, which is normal).
'''

#=========================================================
# Summarization in simple words
#=========================================================
'''
    we fit a reference LinearRegression model on the training set and evaluated MAE/RMSE/R² on the validation set. 
    
    Then we used statsmodels OLS influence diagnostics to identify the top 10 most influential training points by Cook’s Distance. 
    We removed these top 10 rows from training only, retrained LinearRegression, and re-evaluated on validation. 
    
    The validation metrics became dramatically worse (RMSE exploded and R² turned highly negative), 
    showing that these influential points were not harmless noise—removing them destabilized the model. 
    
    We then compared coefficients before vs after removal and 
    observed that some coefficients were relatively stable (especially Latitude/Longitude and MedInc), 
    while others showed large percentage changes (notably Population and AveOccup), 
    often because the original coefficients were close to zero and sign flips make percentage change look huge. 
    
    We also tried Ridge with scaling; at small alphas it behaved almost identically to LinearRegression on validation.
'''