'''
----------------------------------------------------------------------------------------
    ** Understand Missingness (Not all missing data is equal.)

    Type    Meaning                                 Example
    ----    -------                                 -----------
    MCAR    Missing Completely At Random            Network glitch drops a record
    MAR     Missing At Random (but explainable)     Age missing mainly for teens
    MNAR    Missing Not At Random                   Salary missing because people hide high income
----------------------------------------------------------------------------------------
'''

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Detect and Quantify Missingness
datadf = pd.read_csv('./data/week4/day1.csv')

print("\n----------------------\n\n",datadf.info(),"\n\n")
'''
----------------------------------------------------------------------------------------
    Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
    0   PassengerId  120 non-null    int64  
    1   Survived     120 non-null    int64  
    2   Pclass       120 non-null    int64  
    3   Sex          120 non-null    object 
    4   Age          99 non-null     float64
    5   SibSp        120 non-null    int64  
    6   Parch        120 non-null    int64  
    7   Fare         114 non-null    float64
    8   Embarked     120 non-null    object 
    9   Cabin        38 non-null     object 
    10  Ticket       120 non-null    object 
----------------------------------------------------------------------------------------

'''

print("\n----------------------\n\n",datadf.isnull().sum(),"\n\n")
'''
----------------------------------------------------------------------------------------

    PassengerId     0
    Survived        0
    Pclass          0
    Sex             0
    Age            21
    SibSp           0
    Parch           0
    Fare            6
    Embarked        0
    Cabin          82
    Ticket          0
----------------------------------------------------------------------------------------

'''

missing_summary = pd.DataFrame({
    'Missing Values': datadf.isnull().sum(),
    'Missing %': (datadf.isnull().sum() / len(datadf)) * 100
}).sort_values(by='Missing %', ascending=False)

print(missing_summary)
'''
----------------------------------------------------------------------------------------
                Missing Values  Missing %
    Cabin                    82  68.333333
    Age                      21  17.500000
    Fare                      6   5.000000
    Embarked                  4   3.333333
    PassengerId               0   0.000000
    Survived                  0   0.000000
    Pclass                    0   0.000000
    Sex                       0   0.000000
    SibSp                     0   0.000000
    Parch                     0   0.000000
    Ticket                    0   0.000000
----------------------------------------------------------------------------------------

'''
# A heat map that shows missing values for each column
sns.heatmap(datadf.isnull(),cbar=False, cmap='coolwarm')
plt.title('Missing Data Heatmap')
plt.show()

# 2. Impute Missing values

'''
----------------------------------------------------------------------------------------
    Syntax              Type                Shape                       Example
    ------              ----                -------                     ---------
    datadf['Age']       pandas.Series       1-dimensional ((n,))        One column flattened
    datadf[['Age']]     pandas.DataFrame    2-dimensional ((n, 1))      One column but still 2D

    Imputers like SimpleImpurter expects a 2-D array (dataframe) so using a convention of datadf[['Age']] is the correct mechanism
----------------------------------------------------------------------------------------

'''

# a) Numerical Imputation 
num_imput = SimpleImputer(strategy="mean")
datadf[['Age']] = num_imput.fit_transform(datadf[['Age']])
datadf[['Fare']] = num_imput.fit_transform(datadf[['Fare']])

# b) Categorical Imputation
cat_imput = SimpleImputer(strategy="most_frequent")
datadf[['Embarked']] = cat_imput.fit_transform(datadf[['Embarked']])

# c) Constant Filling
const_imput = SimpleImputer(strategy="constant",fill_value='Unknown')
datadf[['Cabin']] = const_imput.fit_transform(datadf[['Cabin']])


'''
----------------------------------------------------------------------------------------
🧩 Rule of thumb:
	•	If < 5 % missing → drop or mean fill is fine.
	•	5–30 % → prefer imputation.
	•	30 % → consider dropping feature unless crucial.
----------------------------------------------------------------------------------------
'''


# 3. Detect Outliers

# a) Using Z-Score
'''
----------------------------------------------------------------------------------------
    A Z-score measures how far a value is from the mean, in units of standard deviation (σ).

    Z = (x - μ)/𝛔

    where
        •	x = the individual value
        •	μ = the mean of the column
        •	𝛔 = the standard deviation of the column

    Why threshold at 3?

    In a normal (bell-curve) distribution, roughly:
        •	68% of values lie within ±1σ
        •	95% within ±2σ
        •	99.7% within ±3σ

    That means only 0.3% of values fall outside ±3σ.
    → Those are extremely unlikely, hence statistical outliers.
----------------------------------------------------------------------------------------

'''

z = np.abs(stats.zscore(datadf['Fare']))
outliers = datadf[z>3]
print(outliers.head())

# b) Using IQR
'''
----------------------------------------------------------------------------------------
What is IQR (Interquartile Range)?

IQR focuses on percentiles, not on mean or standard deviation — 
so it works even when your data is skewed or 
non-normal i.e. when the data isn’t perfectly bell-shaped(like income, house prices, or Titanic fares).

Term                    Meaning
------                  --------
Q1 (25th percentile)    25% of data lies below this value
Q3 (75th percentile)    75% of data lies below this value
IQR = Q3 - Q1           The range of the middle 50% of your data

🔹 Why it works 
    * Instead of assuming a bell-shaped curve, IQR says:
    * “Let’s find the central 50% of data — and if something lies far outside that middle zone, it’s suspicious.”

🔹 The Rule of Thumb
    We mark anything as an outlier if it lies:  
        x < Q1 - 1.5 \times IQR \quad \text{or} \quad x > Q3 + 1.5 \times IQR

    That is, 1.5× the IQR below the 25th percentile or above the 75th percentile.

    You can think of this as the “whisker rule” — it’s what boxplots use to draw whiskers and mark dots (outliers).

🔹 Why 1.5?
    That 1.5 multiplier is a convention, not a law — 
    but it’s a statistically justified one, chosen because it behaves very consistently across data distributions.

🔹 Summary?
Concept         Analogy                 Captures
-------         --------                ---------
Z-score±3σ      For normal data         99.7% range
IQR ±1.5×       For all data            Similar 99% range (without assuming normality)
IQR ±3×         For extreme outliers    99.9%+ range
----------------------------------------------------------------------------------------

'''
Q1 = datadf['Fare'].quantile(0.25)
Q3 = datadf['Fare'].quantile(0.75)
IQR = Q3-Q1
iqroutliers = datadf[(datadf['Fare'] < Q1-1.5*IQR) | (datadf['Fare'] > Q3+1.5*IQR)]
print(iqroutliers.head())

# 4. Visualize Confirmation

# a) Missing value heatmap
# sns.heatmap(datadf.isnull(),cbar=False, cmap='coolwarm')
# plt.title('Missing Data Heatmap')
# plt.show()

# b) Outlier Visulaization
sns.boxenplot(x=datadf['Fare'])
plt.table('Boxplot - Outlier Detection')
plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hands-On Mini Exercise 🧠
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
Dataset: small Titanic sample (5 columns → Age, Fare, Embarked, Cabin, Survived).

Tasks:
	1.	Print missing-value count for each column.
	2.	Impute missing values properly.
	3.	Detect outliers in Fare and Age.
	4.	Plot boxplots before and after cleaning.
	5.	Reflect on:
	•	Which columns had the most missing data?
	•	Did imputation shift the average/median?
	•	How many outliers were removed?
'''

titanicdf = pd.read_csv("./data/week4/titanic.csv")

print("\n\n -- CHECKING MEAN , STD ON DATAFRAME ---- \n\n",titanicdf.describe())

# M.1 Print Missing-value count for each column
titanicmissingdata =pd.DataFrame({
    'datatype' : titanicdf.dtypes,
    'count' : len(titanicdf),
    'Missing Values': titanicdf.isnull().sum(),
    'Missing %': (titanicdf.isnull().sum() / len(titanicdf)) * 100
}).sort_values(by='Missing %', ascending=False)

print(titanicmissingdata)

# M.4.4 a) Plot boxplots before cleaning
fare_before = titanicdf['Fare'].copy()
age_before = titanicdf['Age'].copy()

sns.boxplot(data=titanicdf[['Fare','Age']])
plt.title("Box plot before the cleaning")
plt.show()


# M.2 Impute missing values properly

# Constants Impute
tit_const_impt = SimpleImputer(strategy="constant",fill_value="unknown")
titanicdf[['Cabin']] = tit_const_impt.fit_transform(titanicdf[['Cabin']] )

# Numeric Impute
tit_num_impt = SimpleImputer(strategy="mean")
titanicdf[['Age']] = tit_num_impt.fit_transform(titanicdf[['Age']])
titanicdf[['Fare']] = tit_num_impt.fit_transform(titanicdf[['Fare']])

# Category Impute
tit_cat_impt = SimpleImputer(strategy="most_frequent")
titanicdf[['Embarked']] = tit_cat_impt.fit_transform(titanicdf[['Embarked']])

titanicmissingdata =pd.DataFrame({
    'datatype' : titanicdf.dtypes,
    'count' : len(titanicdf),
    'Missing Values': titanicdf.isnull().sum(),
    'Missing %': (titanicdf.isnull().sum() / len(titanicdf)) * 100
}).sort_values(by='Missing %', ascending=False)

print(titanicmissingdata)

# M.4.4 b) Plot boxplots after cleaning
sns.boxplot(data=titanicdf[['Fare','Age']])
plt.title("Box plot after the cleaning")
plt.show()

print("\n\n -- CHECKING MEAN , STD ON DATAFRAME  AFTER IMPUTATION---- \n\n",titanicdf.describe())

# M.3 Detect outliers in Fare and Age
tz_fare = np.abs(stats.zscore(titanicdf['Fare']))
fareoutlier = titanicdf[tz_fare>3]
print("\n\n ---- FARE OUTLIER ----- \n\n",fareoutlier.head())

tz_age = np.abs(stats.zscore(titanicdf['Age']))
ageoutlier = titanicdf[tz_age>3]
print("\n\n ---- AGE OUTLIER ----- \n\n",ageoutlier.head())

F_Q1 = titanicdf['Fare'].quantile(0.25)
F_Q3 = titanicdf['Fare'].quantile(0.75)
F_IQR = F_Q3-F_Q1
f_filter = ((titanicdf['Fare'] < F_Q1-1.5 * F_IQR) | (titanicdf['Fare'] > F_Q3+1.5 * F_IQR))
print("\n\n ---- FARE OUTLIER USING IQR ----- \n\n",titanicdf[f_filter])

A_Q1 = titanicdf['Age'].quantile(0.25)
A_Q3 = titanicdf['Age'].quantile(0.75)
A_IQR = A_Q3-A_Q1
a_filter = ((titanicdf['Age'] < A_Q1-1.5 * A_IQR) | (titanicdf['Age'] > F_Q3+1.5 * A_IQR))
print("\n\n ---- AGE OUTLIER USING IQR ----- \n\n",titanicdf[a_filter])

# 5.	Reflect on:
# 	•	Which columns had the most missing data? --> Cabin (76%)
# 	•	Did imputation shift the average/median? --> No as we have used mean for missing value ? (standard deviation shifted)
# 	•	How many outliers were removed? (Cabin --> 82, Age --> 21, Fare-->6, Embarked--> 4)