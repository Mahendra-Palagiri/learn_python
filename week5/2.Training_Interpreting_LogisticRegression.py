'''. ---- Week 5 — Day 2: Training & Interpreting Logistic Regression ----
🎯 Learning Goal

By the end of today you will be able to:
	•	Build a preprocessing + logistic regression pipeline
	•	Train a model on your synthetic dataset
	•	Extract and interpret coefficients
	•	Understand how each feature affects survival probability
	•	Validate that model behaviors match the synthetic data’s true structure
'''

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


titdf = pd.read_csv('./data/week5/titanic_synthetic.csv')
# print(titdf.info())
# print(titdf.describe(include='all'))
# print(titdf.head(10))

num_cols = ['Age','Fare']
cat_cols = ['Embarked','Sex']
features = num_cols + cat_cols
target = 'Survived'

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scaler',RobustScaler())
])

cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])


X = titdf[features]
Y = titdf[target]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


final_pipe  = Pipeline([
    ('preprcsr',preprocess),
    ('model',LogisticRegression(max_iter=1000))
])

final_pipe.fit(X_train,Y_train)
finalYpred = final_pipe.predict(X_test)
acc = accuracy_score(Y_test,finalYpred)
# print(acc)


featurenames = final_pipe.named_steps['preprcsr'].get_feature_names_out()
# print(featurenames)


coeff = final_pipe.named_steps['model'].coef_[0]

coeffdf = pd.DataFrame({
    'features': featurenames,
    'coefficients': coeff
}).sort_values(by='coefficients', ascending=False)

print(coeffdf)