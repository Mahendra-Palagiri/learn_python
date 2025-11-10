'''
🎯 Objective

Understand and manage relationships between numerical features so your models stay stable, interpretable, and efficient.

🧩 Concept Building

🧠 1️⃣ Correlation
	•	Definition: Measures how strongly two numeric features move together.
	•	Range = −1 to +1
	•	+1 → perfect positive relationship
	•	−1 → perfect negative relationship
	•	0 → no relationship
	•	Types:
	•	Pearson → linear relationship (default).
	•	Spearman → monotonic (rank-based).
	•	Visualization: seaborn.heatmap(df.corr(), annot=True, cmap='coolwarm')

🧮 2️⃣ Multicollinearity
	•	Happens when two or more features contain the same information.
	•	Consequence: model coefficients become unstable or flip signs (especially in logistic/linear regression).
	•	Distance-based or tree-based models handle it better, but it still hurts interpretability.

📊 3️⃣ Variance Inflation Factor (VIF)
	•	Formula: VIF = 1 / (1 − R²)
	•	Each variable is regressed on the others; a high R² → high VIF.
	•	Rules of thumb:
	•	VIF < 5  → safe
	•	5 ≤ VIF < 10  → watch
	•	VIF ≥ 10  → serious multicollinearity
'''



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


titdf = pd.read_csv("./data/week4/titanic_with_sex.csv")
# print(titdf.info())
# print(titdf.describe(include='all'))

# 🧩 Step 1 — Select numeric columns
numcols = ['Age','Fare']
numdf = titdf[numcols]
# print(numdf.describe(include='all'))


# 📊 Step 2 — Compute and visualize correlation
'''
    ✅ Goal: detect whether columns are strongly related (> 0.8 or < -0.8).
    If you see such pairs, they’re likely redundant.
'''

corr_matrix = numdf.corr()
# print(corr_matrix)
''' ---- output ---
           Age      Fare
Age   1.000000  0.053169
Fare  0.053169  1.000000
'''
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',center=0)
plt.title("Correlation Matrix")
plt.show()



# 🧮 Step 3 — Add a synthetic correlated column
# Now add a feature that is mathematically tied to another (to see how correlation behaves):
titdf['Fare_per_Age'] = titdf['Fare']/titdf['Age']
numcols.append('Fare_per_Age')
numdf = titdf[numcols]
# print(numdf.describe(include='all'))
corr_matrix = numdf.corr()
# print(corr_matrix)
''' ---- output ---
                   Age      Fare  Fare_per_Age
Age           1.000000  0.053169     -0.520776
Fare          0.053169  1.000000     -0.008358
Fare_per_Age -0.520776 -0.008358      1.000000

'''
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',center=0)
plt.title("Correlation Matrix With Synthetic data")
plt.show()

sns.pairplot(numdf[numcols])
plt.title("PairPlot (Relationship between features)")
plt.show()

# 🧩 Step 4 — Calculate Variance Inflation Factor (VIF)
# Now that we’ve visually confirmed relationships between the features, let’s numerically test for multicollinearity using VIF.
X = numdf.dropna()
# print(X.describe(include='all'))

vifdf = pd.DataFrame()
vifdf['Features'] = X.columns
vifdf['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# print(vifdf)
# print(vifdf.describe(include='all'))

# 🧩 Step 5 — Drop redundant features
# Since our goal is to reduce redundant information (not to inflate complexity), we’ll drop the feature with the highest VIF.
numdf = numdf.drop('Fare_per_Age', axis=1)
# print(numdf.describe(include='all'))


# 🧩 Step 7 — Retrain Logistic Regression & Compare Accuracy
# We’ll now train the model again — first with the original features, then with the reduced features, and compare performance.

features = ['Age','Fare','Embarked','Sex']
target = 'Survived'

numcols = ['Age','Fare']
catcols = ['Embarked','Sex']

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scaler',RobustScaler())
])

cat_pipe  = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer([
    ('num',num_pipe,numcols),
    ('catg',cat_pipe,catcols)
])

final_pipe = Pipeline([
    ('preprc',preprocess),
    ('model',LogisticRegression(max_iter=1000))
])

X = titdf[features]
Y = titdf[target]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
final_pipe.fit(X_train,Y_train)
finalypred = final_pipe.predict(X_test)

print("\n\n == ACCURACY SCORE == \n",accuracy_score(Y_test,finalypred))