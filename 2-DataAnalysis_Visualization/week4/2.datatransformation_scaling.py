'''

------------------------------------------------------------------------------------------------------------
🌅 Week 4 – Day 2: Data Transformation & Feature Scaling

🎯 Learning Goals

By the end of today, we’ll be able to:
	1.	Explain why raw numeric features (e.g. Age, Fare, Income, etc.) can distort models if not scaled.
	2.	Apply standardization and normalization using sklearn.preprocessing.
	3.	Understand how scaling affects distance-based models (like KNN or clustering).
	4.	Use Pipeline for safe and reproducible scaling (so you don’t leak data).
	5.	Visualize before/after scaling distributions.
------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------
🧩 Concept Overview

Concept                             What It Means                                           Analogy
-------                             --------------                                          ----------
Standardization                     Transform features so they                              Like shifting and stretching data 
                                    have mean = 0 and std = 1                               onto a “common measuring stick.” 
     
Normalization (Min-Max Scaling)     Scale all values into range [0, 1]                      Like resizing all photos to the same 
                                                                                            frame size before comparison.

RobustScaler                        Uses median & IQR instead of mean/std                   More resistant to outliers.

When to Use                         • KNN, SVM, Logistic Regression, PCA need scaling
                                    • Tree-based models (DecisionTree, RandomForest) don’t.
------------------------------------------------------------------------------------------------------------


'''


import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# # 1. Import the dataset
# titdf = pd.read_csv("./data/week4/titanic.csv")
# print("\n\n------ DATA FRAME INFO --------\n\n")
# print(titdf.info(),"\n\n")

# # 2. Clean numeric columns
# num_cols = ['Fare','Age']
# filt_num_data = titdf[num_cols].dropna() # drop NaN for clean comparison

# print("\n\n -- Min Fare --- > ",filt_num_data['Fare'].min())
# print("\n\n -- Max Fare --- > ",filt_num_data['Fare'].max())
# print("\n\n -- mean Fare --- > ",filt_num_data['Fare'].mean())
# print("\n\n -- median Fare --- > ",filt_num_data['Fare'].median())
# print("\n\n -- std deviation in Fare --- > ",filt_num_data['Fare'].std())
# print("\n\n -- Min Age --- > ",filt_num_data['Age'].min())
# print("\n\n -- Max Age --- > ",filt_num_data['Age'].max())
# print("\n\n -- mean Age --- > ",filt_num_data['Age'].mean())
# print("\n\n -- median Age --- > ",filt_num_data['Age'].median())
# print("\n\n -- std deviation in Age --- > ",filt_num_data['Age'].std())

# # 3. Initialize and set different scalers
# std_scaler = StandardScaler()
# minmax_scaler = MinMaxScaler()
# robust_scaler = RobustScaler()

# X_std = std_scaler.fit_transform(filt_num_data)
# X_minmax = minmax_scaler.fit_transform(filt_num_data)
# X_robust = robust_scaler.fit_transform(filt_num_data)

# # 4. Visualize data
# fig, axes = plt.subplots(1, 4, figsize=(15,5))
# sns.kdeplot(filt_num_data['Fare'], ax=axes[0], label='Original')
# sns.kdeplot(X_std[:,1], ax=axes[1], label='Standardized')
# sns.kdeplot(X_minmax[:,1], ax=axes[2], label='Normalized')
# sns.kdeplot(X_robust[:,1], ax=axes[3], label='Robust')
# for ax in axes: ax.legend()
# plt.show()

# # 5. Optimized and Final Setup
# y = titdf.loc[filt_num_data.index, 'Survived']
# X = filt_num_data  # only ['Fare','Age'] with no NaNs

# preprocess = ColumnTransformer(
#     transformers=[
#     ('age_scaler',StandardScaler(),['Age']),
#     ('fare_scaler',RobustScaler(),['Fare'])
# 	],
#     remainder='drop'
# )

# final_pipe = Pipeline([
#     ('prep', preprocess),
#     ('clf',  LogisticRegression(max_iter=1000, random_state=42))
# ])

# final_pipe.fit(X, y)

# # 1️⃣ Get only the preprocessing part from the pipeline
# scaler = final_pipe.named_steps['prep']

# # 2️⃣ Transform your original X data (Age & Fare)
# X_scaled = scaler.transform(X)

# # 3️⃣ Find out what columns came out of the scaler
# col_names = scaler.get_feature_names_out()
# print(col_names)
# # Example output: ['age_scaler__Age', 'fare_scaler__Fare']

# # 4️⃣ Turn the scaled NumPy array into a small DataFrame
# X_scaled_df = pd.DataFrame(X_scaled, columns=col_names, index=X.index)

# # 5️⃣ Plot using Seaborn
# sns.kdeplot(x=X_scaled_df[col_names[0]], label='Age (scaled)')
# sns.kdeplot(x=X_scaled_df[col_names[1]], label='Fare (scaled)')
# plt.legend()
# plt.title("KDE Plot of Scaled Features")
# plt.show()

'''
	🧩 Mini Challenge

	🧠 Apply StandardScaler and MinMaxScaler to both Age and Fare columns.
	Compare:
		•	Mean & std before and after scaling
		•	Boxplot/KDE shape
		•	How scaling affects outlier influence

	Reflect:
		•	Which scaler best preserved the data’s shape?
		•	Would you use the same scaler for all models?


'''

df = pd.read_csv("./data/week4/titanic.csv")
# print(df.info())
# print(df.head(3))
# print(df.describe(include='all'))

filnumdata = df[['Age','Fare']].dropna()

print("\n\n Fare mean and std deviation before scaling --> ",filnumdata['Fare'].mean(), "\t --- ", filnumdata['Fare'].std(),"\n\n")
print("\n\n Age mean and std deviation before scaling --> ",filnumdata['Age'].mean(), "\t --- ", filnumdata['Age'].std(),"\n\n")

stdscaler = StandardScaler()
mmscaler = MinMaxScaler()

xstd = stdscaler.fit_transform(filnumdata)

# •	xstd[:, 0] → Age (scaled)
# •	xstd[:, 1] → Fare (scaled)

print("\n\n Fare mean and std deviation after Standard scaling --> ",xstd[:, 1].mean(), "\t --- ", xstd[:, 1].std(),"\n\n")
print("\n\n Age mean and std deviation after Standard scaling --> ",xstd[:, 0].mean(), "\t --- ", xstd[:, 0].std(),"\n\n")

xminmax = mmscaler.fit_transform(filnumdata)

print("\n\n Fare mean and std deviation after MinMax scaling --> ",xminmax[:, 1].mean(), "\t --- ", xminmax[:, 1].std(),"\n\n")
print("\n\n Age mean and std deviation after MinMax scaling --> ",xminmax[:, 0].mean(), "\t --- ", xminmax[:, 0].std(),"\n\n")

# FARE
fig, axes = plt.subplots(1, 3, figsize=(15,5))
sns.kdeplot(filnumdata['Fare'], ax=axes[0], label='Original')
sns.kdeplot(xstd[:,1], ax=axes[1], label='Standardized')
sns.kdeplot(xminmax[:,1], ax=axes[2], label='Normalized')
for ax in axes: ax.legend()
plt.show()

# AGE
fig, axes = plt.subplots(1, 3, figsize=(15,5))
sns.kdeplot(filnumdata['Age'], ax=axes[0], label='Original')
sns.kdeplot(xstd[:,0], ax=axes[1], label='Standardized')
sns.kdeplot(xminmax[:,0], ax=axes[2], label='Normalized')
for ax in axes: ax.legend()
plt.show()


'''
--------------------- RETROSPECTION --------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
🧩 1️⃣ Why drop NaN values from numeric columns (instead of imputing)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When we wrote:
X = df[['Age', 'Fare']].dropna()

we are not saying “ignore missing values forever.”
We’re saying: “for the purpose of demonstrating scaling math (z-score, min-max, etc.), let’s temporarily work with complete data.”
Because:


Reason                                    	Explanation
------------								--------------
🧮 Scaling formulas need real numbers		The formula for standardization (x - μ)/σ cannot handle NaN — even one missing value breaks the computation.

🧪 Today’s goal is to see how scaling       We already learned imputation yesterday (Day 1). 
changes numeric distributions, 				Today’s focus is transformation of numeric values, so we use “clean numeric columns” to avoid mixing concepts.
not how to clean data											

⚙️ Scaling should happen after 				In a real ML workflow, you first impute → then scale. 
imputation in a real pipeline				But for a conceptual demo, we skip the imputation part to directly visualize scaling math.

So dropna() here is a convenience for visualization, not a data-cleaning recommendation.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
🧩 2️⃣ Why only Age and Fare and not Survived?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because these are the continuous numeric features in the Titanic dataset.
Here’s how each variable type breaks down:

Column					Type						Nature							Scaling Needed?		 Why / Why Not
--------				-------						----------						---------			 ----------------
Age						Numeric (continuous)		People’s ages (0–80s)			✅ Yes				Different range than other numeric features
Fare					Numeric (continuous)		Ticket prices (0–500+)			✅ Yes				Highly skewed, large variance
Survived				Binary categorical (0/1)	Target label					❌ No				It’s the outcome, not an input feature
Embarked, Sex, Cabin	Categorical (non-numeric)	Port, gender, cabin string		❌ No				Need encoding (like OneHot) before numeric scaling

👉 So only Age and Fare are “true” numeric features suitable for scaling.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
🧩 3️⃣ Why not impute missing numeric values instead of dropping?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, in production or modeling, you always impute before scaling.

So, the correct “real-world” order would be:

1️⃣ Detect missing values
2️⃣ Impute them
3️⃣ Then apply scaling

For example:
```python
	from sklearn.impute import SimpleImputer
	from sklearn.preprocessing import StandardScaler
	from sklearn.pipeline import Pipeline

	num_pipe = Pipeline([
		('impute', SimpleImputer(strategy='mean')),
		('scale', StandardScaler())
	])

	X_scaled = num_pipe.fit_transform(df[['Age', 'Fare']])
```
For our learning , we’re intentionally doing scaling on clean data first to:
	•	Understand how scaling behaves mathematically,
	•	Avoid introducing extra noise from imputation (which can change mean/std),
	•	Keep visualizations simple when we plot before/after scaling.
    

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
🧩 Explain how StandardScaler, MinMaxScaler and RobustScaler differ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

🧩 1️⃣ Why scaling matters in the first place

	Let’s say your dataset has:
	Feature		Typical Range	Example Values
	-------		------------	---------------
	Age			0 – 80			20, 40, 60
	Fare		0 – 500			50, 250, 400

	If you feed this directly into an ML model that uses distance (like KNN, SVM, Logistic Regression),
	the model will think Fare is “more important” than Age — simply because 500 > 80.

	So, scaling brings everything to a common scale before the algorithm computes distances or gradients.

		🧩 When does a model look across columns?

			Only certain models do — specifically, models that measure “distance” or magnitude differences across all features together.

			Let’s take an example to visualize this.

			Suppose you have these two data points (rows in a dataset):
            Passenger	Age	Fare
            --------	--	----√
			A			20	40
			B			40	400
            
            Without scaling:

			The Euclidean distance between A and B is:
						 --------------------------------	
			distance =  √  (20-40)² +  (40-400)²
            so distance is approx ~ --> 360
            
			Even though Age differs by 20 years, that small 20-point difference is completely dwarfed by the Fare difference (360 units).

			So, when your model looks at the overall distance:
				•	99% of that “distance” comes from Fare,
				•	Age is almost ignored.
        
        Summary : Ensure that all the columns are scaled properly so that any models that measure distance doesnt provide wrong results.

🧩 2️⃣ Checking how the scalers work with and wihtout outliers 

⚙️ Step 1: The dataset without outliers

	Let’s start simple:

	Passenger	Age			Fare
	---------	---			-----
	A			20			40
	B			30			60
	C			40			80
	D			50			100

	✅ Everything looks “normal-ish.”
	We can compute scaling comfortably.

    *** 📘 StandardScaler
    
	Z = (x - μ)/𝛔
     
    Mean (μ) and Std (σ):
    Feature		μ		σ
    ------		--		---
	Age			35		12.91
	Fare		70		25.82
    
    Calculate the Z-Score
    Passenger	Age		Age_scaled		Fare	Fare_scaled
    -------		---		---------		---		------------
	A			20		−1.16			40		−1.16
	B			30		−0.39			60		−0.39
	C			40		+0.39			80		+0.39
	D			50		+1.16			100		+1.16

    Perfectly centered. ✅

	*** 📘 MinMaxScaler

	Formula:
	x` = (x - min) / (max - min)
    
    Passenger	Age		Age_scaled	Fare	Fare_scaled
    --------	---		---------	----	-----------
	A			20		0.0			40		0.0
	B			30		0.33		60		0.33
	C			40		0.67		80		0.67
	D			50		1.0			100		1.0

    All values between 0 and 1, perfect spread. ✅


	*** 📘 RobustScaler

	Uses median (Q2) and IQR = Q3 -	Q1.
    Feature		Median (Q2)		Q1		Q3		IQR (Q3−Q1)
    -------		----------		---		--		-----------
	Age			35				27.5	45		17.5
	Fare		70				55		85		30

	Formula:
	x` = (x - Q2)/IQR
	Passenger	Age		Age_scaled	Fare	Fare_scaled
    ---------	---		---------	----	------------
	A			20		−0.86		40		−1.00
	B			30		−0.29		60		−0.33
	C			40		+0.29		80		+0.33
	D			50		+0.86		100		+1.00

	Everything’s stable and clean. ✅
    
⚠️ Step 2: Now, introduce an outlier

	Let’s add Passenger E with a huge Fare = 1000.
    
	Passenger	Age		Fare		
    --------	---		-----
    A			20		40
	B			30		60
	C			40		80
    D			50		100
	E			50		1000 ⟵ OUTLIER!

    Now let’s see what happens.

	🚨 StandardScaler with outlier

	New mean and std for Fare:
		•	Mean = (40 + 60 + 80 + 100 + 1000)/5 = 256
		•	Std ≈ 385

	Apply scaling:
    Passenger	Fare	Scaled
    --------	----	-------
	A			40		(40−256)/385 = −0.56
	B			60		−0.51
	C			80		−0.46	
    D			100		−0.40
	E			1000	+1.93
    
    😬 What happened?
	•	Because of one huge value (1000), the mean and std exploded,
	•	making the other 4 “normal” values appear very close to 0, even though they were spread between 40–100.
	⚠️ StandardScaler gets “distracted” by outliers.
    
    

	🚨 MinMaxScaler with outlier

	Min = 40, Max = 1000
	Formula → (x−40)/(1000−40)
    
    Passenger	Fare	Scaled
    --------	---		------
	A			40		0.0
	B			60		0.02
	C			80		0.04	
    D			100		0.06
	E			1000	1.0

    😬 Now everything except the outlier is squished into 0–0.06 range.
	The model thinks all normal fares are “basically the same.”
	⚠️ MinMaxScaler also suffers badly from outliers.
    
    
    
    ✅ RobustScaler with outlier

	Median (Q2) = 80
	Q1 = 60, Q3 = 100, IQR = 40

	Formula → (x−80)/40
    
	Passenger	Fare	Scaled
    --------	----	------	
	A			40		−1.00
	B			60		−0.50
	C			80		0.00
	D			100		+0.50
	E			1000	+23.00

	✨ Look what happens:
	•	The “normal” range (40–100) still stays between −1 and +1.
	•	The outlier (1000) goes way off (+23) but doesn’t distort everyone else’s scale.
	RobustScaler keeps the central 50% stable.
    
    Summary for Fare :
    Without outlier:
		StandardScaler:   -1.2  -0.4  0.4  1.2
		MinMaxScaler:     0.0   0.3   0.6  1.0
		RobustScaler:    -0.8  -0.3   0.3  0.8

	With outlier (Fare=1000):
		StandardScaler:   -0.56  -0.51  -0.46  -0.40  +1.93
		MinMaxScaler:     0.00   0.02   0.04   0.06   1.00
		RobustScaler:    -1.00  -0.50   0.00   0.50  +23.00
        
    •	StandardScaler and MinMaxScaler “shrink” everything because the outlier distorts their math.
	•	RobustScaler keeps normal values meaningful — it only lets the outlier stretch by itself.
    
    🧩 Step 4: Summary Table
    
    Scaler				Uses			Center		Spread			Outlier Effect		 Good For
    -------				-----			-----		------			--------------		 ----------
	StandardScaler		mean, std		mean=0		std=1			❌ Large impact		Bell-shaped data
	MinMaxScaler		min, max		0–1 range	range-based		❌ Very sensitive	Bounded features
	RobustScaler		median, IQR		median=0	IQR=1			✅ Minimal			Outlier-heavy or skewed data

    ************* A quick rule of thumb to select which scaler to be applied 
    
    Relationship between Mean & Median	Interpretation			Implication
    ----------------------------------	--------------			-------------
	Mean ≈ Median						Symmetric (normal-ish)	✅ StandardScaler fine
	Mean >> Median						Right-skewed			⚠️ Use RobustScaler or log-transform
	Mean << Median						Left-skewed (rare)		⚠️ Use RobustScaler
    
    In our Titanic dataset
    * Min Fare --- >  13.568895885370686
	* Max Fare --- >  250.0
	* Mean Fare --- >  47.20567742127721
	* Median Fare --- >  42.0616928774529
	* Std deviation in Fare --- >  33.53599558985873
    
	* Min Age --- >  0.4
	* Max Age --- >  53.19996122179421
	* Mean Age --- >  26.7872039001908
	* Median Age --- >  26.671436298735237
	* Std deviation in Age --- >  14.18943084053789.  
    
    🧠 Fare:
	•	Mean (47.2) > Median (42.0)
	•	That’s about a 12% gap
	•	Also note: Max = 250, roughly 5× the mean

		✅ Interpretation:
			•	Right-skewed — a few passengers paid extremely high fares
			•	Likely some first-class / luxury ticket outliers

		👉 Use RobustScaler for Fare.
	

	🧠 Age:
		•	Mean (26.8) ≈ Median (26.7)
		•	That’s less than 0.5% difference
		•	Range is small and reasonable: 0.4 to 53 (no massive outlier)

		✅ Interpretation:
			•	Fairly symmetric
			•	No heavy tails

		👉 Use StandardScaler for Age.

    
'''


'''
----------------------- DATA VISUALIZATION REF -------------------

SEABORN + SLICING PLAYBOOK (Quick Reference)

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
df = pd.read_csv(”./data/week4/titanic.csv”)   # has Age, Fare, Embarked, Cabin, Survived, Pclass (if present)

0) First 60 seconds: always do this

	df.info()
	df.head(3)
	df.describe(include=“all”)
	df[‘Embarked’].value_counts(dropna=False)
    
    

1) Common slices (copy/paste patterns)

numeric only

	num = df[[‘Age’,‘Fare’]].dropna()

	boolean filters

	adults = df[df[‘Age’] >= 18]
	cheap = df[df[‘Fare’] < 20]
	survivors = df[df[‘Survived’] == 1]
	q_port = df[df[‘Embarked’] == ‘Q’]

	multiple conditions

	adult_cheap = df[(df[‘Age’]>=18) & (df[‘Fare’]<20)]

	select rows/cols

	subset = df.loc[df[‘Embarked’].isin([‘S’,‘C’]), [‘Age’,‘Fare’,‘Embarked’,‘Survived’]]

	long/melted view for multi-feature plots

	m = df[[‘Age’,‘Fare’,‘Survived’]].melt(id_vars=‘Survived’, var_name=‘Feature’, value_name=‘Value’)
    
    

2) What plot do I use? (mini decision tree)

	Single numeric: sns.histplot or sns.kdeplot

	Numeric vs numeric: sns.scatterplot (+ hue= a category)

	Numeric by category: sns.boxplot, sns.violinplot, or sns.boxenplot

	Category counts: sns.countplot

	Before/after (e.g., scaling): combine melt + sns.kdeplot or side-by-side boxplot
    
    

3) 10 plug-and-play recipes

	A) Distribution of one numeric

		sns.histplot(data=df, x=‘Fare’, bins=30)
		plt.title(“Fare distribution”); plt.show()

	B) KDE with missing dropped

		sns.kdeplot(data=df, x=‘Age’, fill=True)
		plt.title(“Age density”); plt.show()

	C) Numeric by category (outliers pop!)

		sns.boxplot(data=df, x=‘Embarked’, y=‘Fare’)
		sns.stripplot(data=df, x=‘Embarked’, y=‘Fare’, alpha=0.5, color=‘k’, size=2)
		plt.title(“Fare by Embarked”); plt.show()

	D) Two numerics + category hue

		sns.scatterplot(data=df, x=‘Age’, y=‘Fare’, hue=‘Survived’, alpha=0.7)
		plt.title(“Age vs Fare colored by Survived”); plt.show()

	E) Compare scaled vs unscaled (your Day-2 use case)

		suppose you already made X_scaled_df with columns [‘age_scaler__Age’,‘fare_scaler__Fare’]

		orig = df[[‘Age’,‘Fare’]].dropna().assign(Source=‘Original’)
		scaled = pd.DataFrame(X_scaled_df.values, columns=[‘Age’,‘Fare’]).assign(Source=‘Scaled’)
		both = pd.concat([orig, scaled], ignore_index=True)
		sns.kdeplot(data=both, x=‘Age’, hue=‘Source’)
		plt.title(“Age: original vs scaled”); plt.show()
		sns.kdeplot(data=both, x=‘Fare’, hue=‘Source’)
		plt.title(“Fare: original vs scaled”); plt.show()

	F) Facets: same plot split by a category

		g = sns.FacetGrid(df, col=‘Embarked’, col_wrap=3, height=3, sharex=False, sharey=False)
		g.map_dataframe(sns.histplot, x=‘Fare’, bins=20)
		g.fig.suptitle(“Fare by Embarked”, y=1.03); plt.show()

	G) Before/after outlier capping

		after you create df[‘Fare_capped’] = df[‘Fare’].clip(lower, upper)

		m = df[[‘Fare’, ‘Fare_capped’]].melt(var_name=‘Version’, value_name=‘Value’)
		sns.boxenplot(data=m, x=‘Version’, y=‘Value’)
		plt.title(“Fare: original vs capped”); plt.show()

	H) Quick missingness bar

		na = df.isna().sum().reset_index().rename(columns={‘index’:‘col’,0:‘na’})
		sns.barplot(data=na, x=‘na’, y=‘col’)
		plt.title(“Missing values by column”); plt.show()

	I) Log transform for skew (visual)

		import numpy as np
		sns.kdeplot(x=np.log1p(df[‘Fare’]), fill=True)
		plt.title(“log1p(Fare)”); plt.show()

	J) Heatmap of numeric correlations

		sns.heatmap(df[[‘Age’,‘Fare’,‘Survived’]].corr(), annot=True, fmt=’.2f’, cmap=‘vlag’)
		plt.title(“Correlation (numeric)”); plt.show()
		
    

4) Typical “why is my plot blank / error?” fixes

	NaNs: sns.*plot silently drops them. Use .dropna() on the columns you plot.

	Wrong axis: For arrays from scalers (NumPy), use correct column index (X_std[:, 0] for Age if you fit on [Age, Fare]).

	Dtypes: Categories must be object/category; numbers should be float/int.

	df[‘Survived’] = df[‘Survived’].astype(‘category’)
	df[‘Embarked’] = df[‘Embarked’].astype(‘category’)

	Melt vs wide: If you want one axis to say “Feature” and another “Value”, you must melt first.

5) Minimal styling that just works

	sns.set_theme(context=“notebook”, style=“whitegrid”)
	plt.tight_layout()

6) Reusable helpers

	def kde_compare(dfA, dfB, col, labels=(“A”,“B”), title=None):
	sns.kdeplot(dfA[col].dropna(), label=labels[0])
	sns.kdeplot(dfB[col].dropna(), label=labels[1])
	plt.legend()
	if title: plt.title(title)
	plt.show()

	def box_by_cat(df, y, x, title=None):
	sns.boxplot(data=df, x=x, y=y)
	sns.stripplot(data=df, x=x, y=y, color=‘k’, alpha=0.4, size=2)
	if title: plt.title(title)
	plt.show()
'''