'''Week 6 · Day 5 — Outliers, Leverage, Influence (Cook’s Distance)

Topics
	•	What residual “outliers” are vs what “influential” points are
	•	Leverage: unusual X values (feature-space extremes)
	•	Influence: points that can noticeably change our fitted line/model
	•	Cook’s Distance, leverage (hat values), studentized residuals
	•	How we decide what to investigate (not auto-delete rows)

🎯 Learning Goals

By the end of Day 5, we will be able to:
	•	Explain the difference between:
	•	large residuals (badly predicted)
	•	high leverage (unusual inputs)
	•	high influence (changes the model)
	•	Compute and visualize:
	•	Cook’s Distance
	•	leverage (hat values)
	•	studentized residuals
	•	Identify the top influential rows and inspect them safely
	•	Decide what we would try next (robust models, transforms, feature work), without blindly removing data

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# ==========================================================
# Step 1 — Fit an OLS model on our training Set
# ==========================================================

chdf = fetch_california_housing(as_frame=True).frame
'''
print("\n==============================================================================")
print(chdf.info())
'''

target='MedHouseVal'
X = chdf.drop(columns=[target])
Y = chdf[target]

X_trainval,X_test,Y_trainval,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
X_train,X_val,Y_train,Y_val = train_test_split(X_trainval,Y_trainval,test_size=0.2,random_state=42)

X_train_sm = sm.add_constant(X_train)
ols = sm.OLS(Y_train,X_train_sm).fit()

'''
print("\n==============================================================================")
print("R2:",ols.rsquared)
print("Adj R2:",ols.rsquared_adj)

R2: 0.6131488911003316
Adj R2: 0.6129144358828167
'''


# ==========================================================
# Step 2 — Compute influence metrics (Cook’s D, leverage, studentized residuals)
# ==========================================================

influence = ols.get_influence()

# Core diagnostics
cooks_d = influence.cooks_distance[0]          # array
leverage = influence.hat_matrix_diag           # array
stud_resid = influence.resid_studentized_internal  # array


'''
print("\n==============================================================================")
print("\nDiagnostics arrays:")
print("Cook's D shape:", cooks_d.shape)
print("Leverage shape:", leverage.shape)
print("Studentized residual shape:", stud_resid.shape)

Diagnostics arrays:
Cook's D shape: (13209,)
Leverage shape: (13209,)
Studentized residual shape: (13209,)

What each one means (simple)
	•	Studentized residual: “how extreme is the error” after scaling (like a normalized residual)
	•	Leverage: “how unusual is this X row compared to others”
	•	Cook’s D: “if we removed this row, how much would the model change?”

'''

# ==========================================================
# Step 3 — Plot Cook’s Distance (who are the top influencers?)
# ==========================================================
plt.figure()
plt.stem(np.arange(len(cooks_d)), cooks_d)  # stems help us see spikes
plt.xlabel("Row index (train)")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance — Training Set")
plt.savefig("week6/5.1CooksDistancePlot.png")
# plt.show()


# ~~~ Now we'll find the top 10 influential points:
top_k = 10
top_idx = np.argsort(cooks_d)[-top_k:][::-1]

'''
print("\n==============================================================================")
print("Top influential points by Cook's D:")
for rank, idx in enumerate(top_idx, 1):
    print(f"{rank:2d}) idx={idx}  CookD={cooks_d[idx]:.6f}  leverage={leverage[idx]:.6f}  stud_resid={stud_resid[idx]:.3f}")


Top influential points by Cook's D:
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
# Step 4 — Inspect the top influencer points
# ==========================================================
top_rows = X_train.iloc[top_idx].copy()
top_rows["y_train"] = Y_train.iloc[top_idx].values
top_rows["CookD"] = cooks_d[top_idx]
top_rows["Leverage"] = leverage[top_idx]
top_rows["StudResid"] = stud_resid[top_idx]

'''
print("\n==============================================================================")
print("\nTop rows (first 5 shown):")
print(top_rows.head())


Top rows (first 5 shown):
       MedInc  HouseAge   AveRooms  AveBedrms  Population  ...  Longitude  y_train     CookD  Leverage  StudResid
3364   5.5179      36.0   5.142857   1.142857      4198.0  ...    -120.51    0.675  2.001294  0.576413   3.638153
16669  4.2639      46.0   9.076923   1.307692      6532.0  ...    -120.70    3.500  1.904147  0.403434   5.034018
11862  2.6250      25.0  59.875000  15.312500        28.0  ...    -121.25    0.675  0.484712  0.117223  -5.731689
1913   4.0714      19.0  61.812500  11.000000       112.0  ...    -120.06    4.375  0.135771  0.067223   4.117704
1102   2.4028      17.0  31.777778   9.703704        47.0  ...    -121.54    0.675  0.123410  0.054898  -4.372778

[5 rows x 12 columns]

What we’re looking for
	•	Extremely high/low MedInc, rooms/bedrooms ratios, unusual geo coords, etc.
	•	Rows that look “weird” (possible data artifacts) vs “rare but valid” cases
'''

# ==========================================================
# Step 5 — A useful combined diagnostic plot
# ==========================================================
plt.figure()
plt.scatter(leverage, stud_resid, s=20)
plt.xlabel("Leverage")
plt.ylabel("Studentized Residual")
plt.title("Leverage vs Studentized Residual (Training)")
plt.axhline(0)
plt.savefig("week6/5.2Leverage_StudentizedResidual.png")
# plt.show()

'''
Interpretation:
	•	high leverage + large |studentized residual| are the most suspicious
	•	Cook’s D helps quantify overall influence
'''


# ==========================================================
# RETROSPECTION
# ==========================================================
'''
Understand more about the metrics
        * How its calculated
        * What is the expected value
        * What values are considered as deviations and how to measure. (Close, too big, too small)

    A) Stundentized Residual
        
        Let’s make it concrete with one row

        Pick any index i (we’ll use the “top Cook’s D” indices soon). For that i:

        A) Studentized residual (error extremeness)
            •	residual = pred - actual
            •	studentized residual = residual divided by an estimate of its standard deviation

        So it’s like a z-score for errors:
            •	stud_resid = 0 → prediction is basically on target
            •	stud_resid = 2.5 → about 2.5 “standard error units” off (pretty large)
            •	stud_resid = -3.0 → very underpredicted

        Rule of thumb:
            •	|stud_resid| > 2 is worth looking at
            •	|stud_resid| > 3 is very suspicious

    B) Leverage (X unusualness)

        Leverage is: “is this row’s X far from the center of the X cloud?”
            •	high leverage means the feature combination is rare/extreme (unusual latitude/longitude, extreme MedInc, extreme room ratios, etc.)

        Rule of thumb:
            •	average leverage ≈ (p + 1) / n
        where p = number of features, +1 for intercept
        Here p=8 (we have 8 features in our california housing dataset),
        so average ≈ 9/13209 ≈ 0.00068
            •	values several times larger than that (like >0.005 or >0.01) are “high” relative to the average

    C) Cook’s D (overall influence)

        Cook’s D combines both:
            •	high leverage (unusual X)
            •	and/or high residual (big error)

        Cook’s D answers:

        “If we remove this row and refit, how much would the fitted model change overall?”

        Rule of thumb:
            •	CookD > 4/n is often used as a “flag”
        here 4/13209 ≈ 0.00030
        (not a hard rule — just a trigger to inspect)

⸻

    Why these three are different (simple scenarios)

    Scenario 1: Big error but normal X
        •	high |stud_resid|
        •	low/moderate leverage
        •	CookD might be moderate
    Meaning: the model struggles on this case, but removing it won’t change the model much.

    Scenario 2: Weird X but model predicts it well
        •	low |stud_resid|
        •	high leverage
        •	CookD can still be noticeable
    Meaning: it’s an extreme input. Even if it fits well, it can “anchor” the line.

    Scenario 3: Weird X AND big error (most dangerous)
        •	high |stud_resid|
        •	high leverage
        •	high CookD
    Meaning: this row can pull the model and distort coefficients.

⸻

    Top influential points by Cook's D:
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

    For each top row, we’ll classify it like:
	•	“High CookD because leverage is huge”
	•	“High CookD because residual is huge”
	•	“Both → very influential”

    We’ll read each row like:
	•	Leverage: “how unusual are the inputs (X)?”
	•	Studentized residual: “how big is the error (standardized)?”
	•	Cook’s D: “overall influence = (unusual X) + (big error) → how much the model can change if we remove it”

    1) idx=2757 — CookD=2.00, leverage=0.576, stud_resid=3.638

        Diagnosis: extreme leverage + large error → very influential
        This is the most dangerous type:
            •	unusual X and wrong prediction
            •	can pull coefficients strongly

    2) idx=2580 — CookD=1.90, leverage=0.403, stud_resid=5.034

        Diagnosis: extreme leverage + very large error → very influential
        Same as #1, even more error.

    ✅ These two are “red alert” points.

    ⸻

    3) idx=225 — CookD=0.485, leverage=0.117, stud_resid=-5.732

        Diagnosis: high leverage + very large error (underprediction)
        Less leverage than #1/#2 but still insanely high vs normal.

    ⸻

    4) idx=10876 — CookD=0.136, leverage=0.067, stud_resid=4.118

        Diagnosis: moderately high leverage + large error
        Still influential.

    5) idx=3798 — CookD=0.123, leverage=0.055, stud_resid=-4.373

     Diagnosis: moderately high leverage + large error

    6) idx=9659 — CookD=0.051, leverage=0.052, stud_resid=2.911

        Diagnosis: high leverage + moderate-to-large error
        Influential mainly because leverage is high.

    ⸻

    7) idx=1080 — CookD=0.0375, leverage=0.00483, stud_resid=-8.343

        Diagnosis: huge error but leverage is not extreme (still > average, but not crazy).
        This is the “big residual” type:
            •	model is very wrong for this point
            •	but because X isn’t super unusual, removing it won’t rotate the whole model as much as #1/#2.

    Still worth inspecting because |stud_resid| = 8.3 is enormous.

    ⸻

    8) idx=4023 — CookD=0.0369, leverage=0.0337, stud_resid=-3.085

        Diagnosis: moderate leverage + large error

    9) idx=10332 — CookD=0.0367, leverage=0.0331, stud_resid=-3.106

        Diagnosis: moderate leverage + large error

    10) idx=2308 — CookD=0.0262, leverage=0.0476, stud_resid=-2.171

        Diagnosis: higher leverage + moderate error
        Influence mostly from leverage.

    ⸻

    Summary diagnosis (what we learned today)

    What’s driving Cook’s D in our data?

    Mostly leverage.

    Look at #1 and #2:
        •	leverage 0.576 and 0.403 are unbelievably high.
    That’s why CookD is huge.

    What does that mean?

    We likely have a handful of training rows with extreme feature combinations that are not typical of the dataset. 
    They can distort the fitted coefficients

 np.argsort(cooks_d)[-top_k:][::-1] -->  what we are doing here

    What we want

        We want the indices (row numbers) of the top_k largest Cook’s Distance values — i.e., the most influential points.

        cooks_d is an array like:
            •	cooks_d[0] = Cook’s D for row 0
            •	cooks_d[1] = Cook’s D for row 1
            •	…
        So we need the row indices of the biggest values.

    ⸻

    Step 1: np.argsort(cooks_d)

    argsort does not return sorted values.
    It returns the indices that would sort the array.

    example:
        cooks_d = [0.2, 0.05, 1.3, 0.4]
        np.argsort(cooks_d)  -> [1, 0, 3, 2]

    Why?
	•	cooks_d[1] = 0.05 (smallest)
	•	cooks_d[0] = 0.2
	•	cooks_d[3] = 0.4
	•	cooks_d[2] = 1.3 (largest)

    So after argsort, we have indices in ascending order of Cook’s D.

    ⸻

    Step 2: [-top_k:]

    This slices the last top_k indices from that sorted list.

    Since the list is in ascending order:
        •	the last top_k correspond to the largest Cook’s D values

    So now we have “the indices of the biggest Cook’s D rows”, but still in ascending order.

    ⸻

    Step 3: [::-1]

    This reverses the slice.

    So now we have the indices in descending order:
        •	biggest Cook’s D first
        •	then second biggest
        •	etc.

    ⸻

    Final result

    top_idx becomes something like:
    [2757, 2580, 225, 10876, ...]

    These are the row indices in the training set with the highest influence.
    
'''


#=========================================================
# Summarization in simple words
#=========================================================
'''
    We used statsmodels OLS and explicitly added the intercept using sm.add_constant(X_train), then used ols.get_influence() to compute influence diagnostics. 
    
    From the influence output we examined three metrics: 
        * studentized residuals, 
        * leverage (hat values), and 
        * Cook’s Distance. 
        
    Studentized residuals are residuals scaled by their estimated standard deviation (a standardized error), 

    where |SR|>2 is worth inspecting and |SR|>3 is very large. 
    Average leverage is roughly (p+1)/n (with p features and n rows), and 
    a common Cook’s D flag is 4/n. 
    
    We ranked the top 10 most influential points by Cook’s D and saw that 
    * high influence typically comes from high leverage (unusual X), large standardized residuals (big error), or both. 
    * We learned that “high Cook’s D” means a point can significantly affect model parameters,
      but it doesn’t automatically mean the point is wrong or should be removed.
'''