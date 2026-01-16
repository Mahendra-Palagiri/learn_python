''' Week 6 · Day 3 — Statsmodels OLS + Interpreting Coefficients

Topics
	•	Why we use Statsmodels OLS (interpretability-first)
	•	How OLS differs from scikit-learn (adds stats + inference)
	•	Reading the OLS output:
	•	coefficients, intercept
	•	confidence intervals
	•	p-values (and where we can misuse them)
	•	R^2 vs Adjusted R^2
	•	Writing human explanations:
	•	“If X increases by 1 unit, predicted Y changes by ___ (holding others constant)”

🎯 Learning Goals

By the end of Day 3, we will be able to:
	•	Fit an OLS regression using statsmodels
	•	Explain what a coefficient means in plain language
	•	Use confidence intervals to express uncertainty
	•	Understand what p-values are trying to tell us (and their limitations)
	•	Compare R^2 vs Adjusted R^2 and explain why adjusted matters
'''

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# 1️⃣ ~~ Fetch Dataset 
cfhdf = fetch_california_housing(as_frame=True).frame

# print(cfhdf.info())
'''
	RangeIndex: 20640 entries, 0 to 20639
	Data columns (total 9 columns):
	#   Column       Non-Null Count  Dtype  
	---  ------       --------------  -----  
	0   MedInc       20640 non-null  float64
	1   HouseAge     20640 non-null  float64
	2   AveRooms     20640 non-null  float64
	3   AveBedrms    20640 non-null  float64
	4   Population   20640 non-null  float64
	5   AveOccup     20640 non-null  float64
	6   Latitude     20640 non-null  float64
	7   Longitude    20640 non-null  float64
	8   MedHouseVal  20640 non-null  float64
'''


# 2️⃣ ~~ Prepare for train, validate and test
target = 'MedHouseVal'
X = cfhdf.drop(columns=[target])
Y = cfhdf[target]

X_trainval,X_test,Y_trainval,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
X_train,X_Val,Y_train,Y_val = train_test_split(X_trainval,Y_trainval,test_size=0.2,random_state=42)


# 3️⃣ ~~ Introducting Stats Models.
# Stats Models doesnt add intercept by default. (have to explicitly add)
X_train_sm = sm.add_constant(X_train) #  --> This is our intercept

ols = sm.OLS(Y_train,X_train_sm).fit()
# print(ols.summary())

'''
	OLS Regression Results                            
	==============================================================================
	Dep. Variable:            MedHouseVal   R-squared:                       0.613
	Model:                            OLS   Adj. R-squared:                  0.613
	Method:                 Least Squares   F-statistic:                     2615.
	Date:                Fri, 16 Jan 2026   Prob (F-statistic):               0.00
	Time:                        06:17:46   Log-Likelihood:                -14333.
	No. Observations:               13209   AIC:                         2.868e+04
	Df Residuals:                   13200   BIC:                         2.875e+04
	Df Model:                           8                                         
	Covariance Type:            nonrobust                                         
	==============================================================================
					coef    std err          t      P>|t|      [0.025      0.975]
	------------------------------------------------------------------------------
	const        -36.5885      0.818    -44.740      0.000     -38.191     -34.985
	MedInc         0.4517      0.005     84.520      0.000       0.441       0.462
	HouseAge       0.0096      0.001     17.428      0.000       0.009       0.011
	AveRooms      -0.1259      0.008    -16.676      0.000      -0.141      -0.111
	AveBedrms      0.7815      0.037     20.989      0.000       0.709       0.855
	Population -7.343e-06   5.84e-06     -1.258      0.209   -1.88e-05     4.1e-06
	AveOccup      -0.0040      0.001     -4.354      0.000      -0.006      -0.002
	Latitude      -0.4169      0.009    -46.404      0.000      -0.434      -0.399
	Longitude     -0.4293      0.009    -45.891      0.000      -0.448      -0.411
	==============================================================================
	Omnibus:                     2706.643   Durbin-Watson:                   1.968
	Prob(Omnibus):                  0.000   Jarque-Bera (JB):             7960.517
	Skew:                           1.070   Prob(JB):                         0.00
	Kurtosis:                       6.144   Cond. No.                     2.40e+05
	==============================================================================

	Notes:
	[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
	[2] The condition number is large, 2.4e+05. This might indicate that there are
	strong multicollinearity or other numerical problems.
'''


# 4️⃣ ~~~ Extract the important parts we actually use ~~~
params = ols.params
conf = ols.conf_int()
pvals = ols.pvalues

# print("\n==============================================================================")
# print("Top coefficients (raw):")
# print(params.sort_values(key=np.abs, ascending=False).head(10))

# print("\n==============================================================================")
# print("95% Confidence intervals (first 5):")
# print(conf.head())

# print("\n==============================================================================")
# print("P-values (first 5):")
# print(pvals.head())

# print("\n==============================================================================")
# print("R2:", ols.rsquared)
# print("Adj R2:", ols.rsquared_adj,"\n")

'''
	What we look for today:
	•	Which coefficients are large in magnitude?
	•	Are their confidence intervals tight or wide?
	•	Does adjusted R² differ meaningfully from R²?
    
	==============================================================================
	Top coefficients (raw):
	const        -36.588488
	AveBedrms      0.781543
	MedInc         0.451677
	Longitude     -0.429343
	Latitude      -0.416855
	AveRooms      -0.125858
	HouseAge       0.009612
	AveOccup      -0.003971
	Population    -0.000007
	dtype: float64

	==============================================================================
	95% Confidence intervals (first 5):
					0          1
	const     -38.191497 -34.985480
	MedInc      0.441202   0.462152
	HouseAge    0.008531   0.010693
	AveRooms   -0.140651  -0.111064
	AveBedrms   0.708556   0.854530

	==============================================================================
	P-values (first 5):
	const        0.000000e+00
	MedInc       0.000000e+00
	HouseAge     2.874689e-67
	AveRooms     8.331727e-62
	AveBedrms    3.055961e-96
	dtype: float64

	==============================================================================
	R2: 0.6131488911003316
	Adj R2: 0.6129144358828167
'''


# 5️⃣ ~~~ Evaluate data for top 3 features
# “Holding other variables constant, a 1-unit increase in  is associated with a  change in predicted MedHouseVal.”
def interpret(feature_name: str):
    coef = params[feature_name]
    lo, hi = conf.loc[feature_name]
    print("\n==============================================================================")
    print(f"Interpretation for {feature_name}:")
    print(f"  Coef: {coef:.4f}")
    print(f"  95% CI: [{lo:.4f}, {hi:.4f}]")
    print(f"  Holding other variables constant, +1 in {feature_name} changes predicted {target} by {coef:.4f}.")

for f in ["MedInc", "HouseAge", "AveRooms"]:
    if f in params.index:
        interpret(f)
        
'''
	==============================================================================
	Interpretation for MedInc:
	Coef: 0.4517
	95% CI: [0.4412, 0.4622]
	Holding other variables constant, +1 in MedInc changes predicted MedHouseVal by 0.4517.

	==============================================================================
	Interpretation for HouseAge:
	Coef: 0.0096
	95% CI: [0.0085, 0.0107]
	Holding other variables constant, +1 in HouseAge changes predicted MedHouseVal by 0.0096.

	==============================================================================
	Interpretation for AveRooms:
	Coef: -0.1259
	95% CI: [-0.1407, -0.1111]
	Holding other variables constant, +1 in AveRooms changes predicted MedHouseVal by -0.1259.
'''


# 6️⃣ ~~~   Important caution (p-values)

# Today we do not treat p-values as “truth”. We use them carefully:
# 	•	p-values depend on assumptions (independent errors, correct model form, etc.)
# 	•	correlated features can make p-values misleading
# 	•	huge datasets can make tiny effects “statistically significant” but practically small


''' ----------- Retrospection ----------------
Q) 
* what is statsmodels, 
* why are we using here. 
* what is intercept 
* what do we mean statsmodels doesnt automatically add incerpt like sklearn does.

	Statsmodels vs scikit-learn (simple + practical)

	1) What is statsmodels?

		Statsmodels is a Python library for “classical statistics models.”

		If scikit-learn is mainly about:
			•	“build a model that predicts well”

		Statsmodels is mainly about:
			•	“build a model and understand it”
			•	“show me the equation, the coefficients, and the uncertainty”
			•	“give me statistical details like confidence intervals and p-values”

		So we use statsmodels in Day 3 because Day 3 is about interpretation and model explanation, not tuning performance.

	⸻

	Why are we using it here (Week 6 Regression Deep Dive)?

		Day 2 proved:
			•	our regression workflow works
			•	a simple model beats baseline

		Day 3 focuses on:
			•	what the model learned
			•	what each feature is doing
			•	how sure we are about those effects (confidence intervals)
			•	how to read regression output like a pro

		Statsmodels is built exactly for this.

	⸻

	2) What is the intercept?

		The intercept is just the starting point of our line (or plane).

		With one feature:
		y(pred) = w x + b
			•	b is the intercept

		With many features:
		y(pred) = b + w_1 x_1 + w_2 x_2 + ..... + w_k x_k

		Meaning of intercept (simple):
			•	It’s the predicted value when all features are 0.

		Even if “all features = 0” isn’t realistic, the intercept is still useful because it lets the model shift up/down to best match the data.

		Without an intercept, the model is forced to go through the origin (0,0) in the feature space, which is often wrong.

	⸻

	3) “Statsmodels doesn’t automatically add intercept like sklearn”

		This sounds confusing but it’s simple.

		scikit-learn behavior

		In LinearRegression(), scikit-learn includes an intercept by default:
			•	it learns b automatically (fit_intercept=True by default)

		So we can just do:
		lr.fit(X, y)
        
        and it learns both slopes and intercept.

		statsmodels behavior

		Statsmodels expects you to be explicit:
			•	it only fits what you put in X

		If you pass X with only your features, statsmodels fits:
		y(pred)= w_1 x_1 + w_2 x_2 + ....
		(no + b)

		To include intercept b, we must add a column of 1s to X:
			•	because b x 1 is just b

		That’s exactly what this does:
        X_train_sm = sm.add_constant(X_train)
        
        It adds a new column (usually called const) filled with 1s.

		Then the model becomes:
		y(pred) = (const coef)x 1 + w_1 x_1 + ...
		and the “const coef” is the intercept.
        
        Tiny example (so it’s visual)

		If our X is:
        x
        ---
		2
		5
        
        Adding constant makes it:
        const		x
        -----		---
		1			2
		1			5
        
        Now the model can learn:
		\hat{y} = b . 1 + w .  x
		which is the normal line.

		⸻

		Why we like statsmodels output (what we get that sklearn doesn’t give easily)

		Statsmodels gives a full report including:
			•	coefficient values
			•	confidence intervals (range of plausible values)
			•	p-values (statistical significance; we will treat carefully)
			•	R^2 and adjusted R^2

		Sklearn is more “engineering/prediction” oriented; statsmodels is more “analysis/explanation” oriented.

    
Q) What does OLS stand for?

	OLS = Ordinary Least Squares

	OLS is the most common way to fit linear regression.

	What it means in simple terms
	We have a line (or plane with many features):

	\hat{y} = b + w_1x_1 + w_2x_2 + \dots

	For every data point, there’s an error:

	\text{error} = \hat{y} - y

	OLS chooses the weights and intercept that make the total error as small as possible, using this specific rule:

	Square the errors and add them up, then pick the model that makes that sum smallest.

	That’s the “least squares” part:
		•	least = smallest
		•	squares = squared errors

	“Ordinary” just means the standard version (not weighted, not robust, etc.).

	Why we care about OLS in Day 3
	Because statsmodels OLS doesn’t just give predictions — it gives us a full statistical report:
		•	coefficients
		•	intercept
		•	confidence intervals
		•	p-values
		•	R^2, adjusted R^2

	So Day 3 is about understanding what the linear regression learned, and OLS is the classic way to do it.

	If you want a 1-line intuition: OLS finds the line that best fits the data by minimizing squared mistakes.

'''