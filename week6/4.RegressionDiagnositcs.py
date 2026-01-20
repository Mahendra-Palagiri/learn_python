'''Week 6 · Day 4 — Regression Diagnostics (Residuals, Non-Linearity, Heteroskedasticity)

Topics
	•	What residuals are and why diagnostics matter
	•	Residual plots to detect:
	•	non-linearity (curvature)
	•	heteroskedasticity (fan / funnel shape)
	•	outliers (extreme residuals)
	•	“Good vs suspicious” diagnostic patterns
	•	Quick tests (optional): Breusch–Pagan for heteroskedasticity

🎯 Learning Goals

By the end of Day 4, we will be able to:
	•	Generate residual diagnostics plots correctly
	•	Recognize common failure patterns (curve, fan, clusters)
	•	Explain what each pattern suggests we should try next
	•	Understand why “pretty R^2” is not enough

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import het_breuschpagan


# ==========================================================
# Step 1 — Fit a model we will diagnose
# ==========================================================
# 1️⃣ Load and Split data
chdf = fetch_california_housing(as_frame=True).frame

# print(chdf.info())

target = 'MedHouseVal'
X = chdf.drop(columns=[target])
Y = chdf[target]

X_trainval,X_test,Y_trainval,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
X_train,X_val,Y_train,Y_val = train_test_split(X_trainval,Y_trainval,test_size=0.2,random_state=42)

# 2️⃣ Fit Model
lr = LinearRegression()
lr.fit(X_train,Y_train)

# 3️⃣ Predict based on validation dataset
yval_pred = lr.predict(X_val)
residuals = yval_pred-Y_val
rmse = np.sqrt(mean_squared_error(Y_val,yval_pred))

print("\n==============================================================================")
print("Validation RMSE:",rmse,"\n")

# ==========================================================
# Step 2 — Diagnostic plot set
# ==========================================================

# Plot A: Residuals vs Predicted
plt.figure()
plt.scatter(yval_pred,residuals)
plt.axhline(0)
plt.xlabel("Predicted Y (Val)")
plt.ylabel("Residucals (Val)")
plt.title("Predicted vs Residuals -- Validation Set")
# plt.show()

'''
What we want: random cloud centered around 0
Suspicious: curve or fan shape
'''

# Plot B: Residuals vs Actual. (Useful when predicted values are clipped or bunched.)
plt.figure()
plt.scatter(Y_val,residuals)
plt.axhline(0)
plt.xlabel("Actual Y (Val)")
plt.ylabel("Residuals (Val)")
plt.title("Acutal vs Residual  --> Validation Set")
# plt.show()


# Plot C: Predicted vs Actual with diagonal
plt.figure()
plt.scatter(Y_val,yval_pred)
min_y = min(Y_val.min(), yval_pred.min())
max_y = max(Y_val.max(), yval_pred.max())
plt.plot([min_y,max_y],[min_y,max_y])
plt.xlabel("Actual (Val)")
plt.ylabel("Predicted (Val)")
plt.title("Actual vs Predicted --> Validation Set")
# plt.show()

# Plot D: Residual histogram (distribution sanity)
plt.figure()
plt.hist(residuals,bins=50)
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residual Distribution --> Validation Set")
# plt.show()


# ==========================================================
# Step 3 :: Diagnostic plot set (ALL IN ONE FIGURE)
# ==========================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot A: Residuals vs Predicted
ax = axes[0, 0]
ax.scatter(yval_pred, residuals, alpha=0.6)
ax.axhline(0)
ax.set_xlabel("Predicted Y (Val)")
ax.set_ylabel("Residuals (Val)")
ax.set_title("Predicted vs Residuals — Validation Set")

# Plot B: Residuals vs Actual
ax = axes[0, 1]
ax.scatter(Y_val, residuals, alpha=0.6)
ax.axhline(0)
ax.set_xlabel("Actual Y (Val)")
ax.set_ylabel("Residuals (Val)")
ax.set_title("Actual vs Residuals — Validation Set")

# Plot C: Predicted vs Actual with diagonal
ax = axes[1, 0]
ax.scatter(Y_val, yval_pred, alpha=0.6)
min_y = min(Y_val.min(), yval_pred.min())
max_y = max(Y_val.max(), yval_pred.max())
ax.plot([min_y, max_y], [min_y, max_y])
ax.set_xlabel("Actual (Val)")
ax.set_ylabel("Predicted (Val)")
ax.set_title("Actual vs Predicted — Validation Set")

# Plot D: Residual histogram
ax = axes[1, 1]
ax.hist(residuals, bins=50)
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution — Validation Set")

plt.tight_layout()
fig.savefig("week6/4.1Regression_diagnostics.png", dpi=200, bbox_inches="tight")
# plt.show()

"""
![Diagnostics](week6_day4_regression_diagnostics.png)
"""

# ==========================================================
# Step 4 :: Breusch–Pagan test for heteroskedasticity
# ==========================================================

X_val_sm = sm.add_constant(X_val)
ols = sm.OLS(Y_val,X_val_sm).fit()

bp = het_breuschpagan(ols.resid,X_val_sm)
labels = ["LM Stat", "LM p-value", "F Stat", "F p-value"]

print("\nBreusch–Pagan test:")
for name, val in zip(labels, bp):
    print(name, ":", val)

'''
Breusch–Pagan test:
LM Stat : 300.6642950412045
LM p-value : 2.9735644217256013e-60
F Stat : 41.234070953073186
F p-value : 3.1039460445456923e-63

Interpretation (simple):
	•	low p-value suggests heteroskedasticity is present
'''

''' ---- RETROSPECTION -----
What patterns mean (simple mapping)

**** If we see a “curve” in residuals vs predicted

        Likely:
            •	non-linear relationship we didn’t model
        Next try:
            •	polynomial features
            •	interactions
            •	log transforms

**** If we see a “fan” / “funnel”

        Likely:
            •	heteroskedasticity (variance changes with prediction level)
        Next try:
            •	transform y (like log)
            •	robust regression / robust SE (later)
            •	different model class

**** If we see extreme points far away

        Likely:
            •	outliers or influential points
        Next try:
            •	examine those rows
            •	Cook’s distance (Day 5)


# EXPLANATION ON THE GRAPH (4.1Regression_diagnostics.png)
1) Predicted vs Residuals (top-left): why that sharp diagonal “edge”?

We defined residuals as:

\text{residual} = \hat{y} - y

In the California Housing dataset, the target MedHouseVal is capped at 5.0 (in $100k units). So:

y \le 5

That means residuals must satisfy:

\text{residual} = \hat{y} - y \ge \hat{y} - 5

So there’s a hard lower boundary line:

\text{residual} = \hat{y} - 5

✅ That is exactly the sharp diagonal “floor” you’re seeing.
Points can’t fall below it because the true y can’t exceed 5.

Interpretation: This is not “a model bug.” It’s a dataset label cap creating a geometric constraint in residual space.

⸻

2) Actual vs Residuals (top-right): why the vertical band at Actual = 5?

Same reason: many samples have Actual y = 5 (hit the cap).
So we see a vertical wall at x = 5.

Also notice the residuals near actual=5 spread a lot — that’s telling us:
	•	for capped cases, the model can predict above/below 5, but the true label is stuck at 5
	•	residuals there can look systematically biased because the data is censored.

⸻

3) Actual vs Predicted (bottom-left): why the tall vertical column at Actual = 5?

Again: lots of points have Actual=5, so we get a vertical stack.

Also, notice we sometimes predict above 5 (up to ~7+). That’s normal for plain linear regression because it doesn’t “know” the label cap. It will happily extrapolate beyond 5.

⸻

4) Residual distribution (bottom-right): what do we see?

The histogram is:
	•	roughly centered near 0 (good)
	•	but not perfectly symmetric, and there’s a heavier tail (common in real-world regression)

With this dataset, we should expect residual weirdness because:
	•	the target is censored at 5 (so errors can’t behave like a clean bell curve),
	•	and the true relationship is not perfectly linear.

⸻

What does this mean for Day 4 diagnostics?

What looks “good”
	•	We do see a reasonable diagonal trend in Actual vs Predicted.
	•	Residuals are broadly around 0 for a big chunk of the data.

What looks “suspicious / informative”
	•	The cap-driven boundary (diagonal floor + vertical wall) is telling us our target is censored.
	•	The spread of residuals varies with prediction/actual in places → likely heteroskedasticity (variance not constant).
	•	There may also be some non-linearity (common with housing).

⸻

What we should do next (our Day 4 conclusions → Day 5 direction)

Here are the best next moves, in order:
	1.	Acknowledge censoring
	•	Our diagnostics are partly distorted by the y=5 cap.
	•	Linear regression isn’t ideal for censored targets.
	2.	Run a quick heteroskedasticity check (optional but useful)
	•	Breusch–Pagan test (you already have the code pattern from earlier days).
	•	Even if it flags heteroskedasticity, the plots already suggest it.
	3.	Plan improvements we can try (later days)
	•	Transform target: try log1p(y) (helps variance issues sometimes, though the cap complicates it).
	•	Try a model that handles non-linearities better (Decision Tree / Random Forest / Gradient Boosting).
	•	Consider methods designed for censoring (advanced topic): Tobit / censored regression.
	4.	Day 5 focus
	•	Outliers + leverage + influence (Cook’s distance)
	•	Identify points with extreme residuals and see if they’re legitimate or data quirks.

    
Interpolate "What patterns mean (simple mapping)" explanation with the charts and help diganoze the shape  (Predicted vs Residuals) i dont either see a clear curve or fan/funnel shape
mapping, applied to our charts

What we’re looking at
	•	x-axis: Predicted \hat{y}
	•	y-axis: Residual (\hat{y} - y)
	•	A “perfect” world would look like a random cloud centered around 0, with roughly the same vertical spread everywhere.

⸻

1) The most obvious shape: the sharp diagonal “edge”

In our plot there’s a clean diagonal line (a hard boundary). That is not a curve or a funnel.

✅ Diagnosis: this is caused by the target cap at y = 5 (censoring).

Why:
	•	residual = pred − actual
	•	actual can’t exceed 5
	•	so residual can’t be less than (pred − 5)
That creates the diagonal boundary.

What this means in our mapping:
This isn’t “non-linearity” or “heteroskedasticity” by itself — it’s a data constraint that shapes residuals.

⸻

2) Do we see a curve?

A curve would look like the residual cloud bending:
	•	residuals mostly positive at low predictions and mostly negative at mid predictions, etc.
	•	like a smile or frown pattern around the 0 line

In our plot, the big cloud (predictions roughly 0–4) is not clearly bending into a smooth curve. It looks more like a blob around zero.

✅ Diagnosis: We do not have strong visual evidence of a classic “curve” pattern here.

So we can’t confidently conclude “non-linearity” just from this plot alone.

(Non-linearity may still exist, but this plot is dominated by the cap boundary.)

⸻

3) Do we see a fan/funnel (heteroskedasticity)?

A funnel would look like:
	•	narrow spread at low predicted values
	•	wider spread as predicted increases (or the reverse)

In our plot, it’s tricky because:
	•	the cap boundary creates a wedge-like region for high predicted values (>5)
	•	that wedge can look like “fanning,” but it’s not variance growth — it’s geometry from clipping

So we should focus on the main blob region (pred ~0 to 4):
	•	the vertical spread there looks fairly similar, not a clear “opening funnel.”

✅ Diagnosis: We do not see a clean, textbook funnel shape in the main region.

⸻

4) What shape do we actually see?

We see two regimes:

Regime A: predicted ~0 to 4
	•	looks like a somewhat thick cloud around 0
	•	no strong curve, no strong fan

Regime B: predicted > ~5
	•	points line up near the diagonal boundary
	•	this is because many true values are capped at 5
	•	residuals here are forced by censoring, not just model variance

✅ Our main diagnosis for Predicted vs Residuals:

The plot is heavily influenced by label capping at 5, which creates the sharp diagonal boundary and makes “fan/curve” detection harder. In the uncapped region (pred 0–4), residuals look reasonably cloud-like without a strong curve or funnel.

That’s a good, honest diagnosis.
'''