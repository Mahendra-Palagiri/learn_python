'''
Week 5 — Day 3: Classification Metrics & Confusion Matrix

# 🎯 Learning Goal

By the end of today we will:
	•	Understand the confusion matrix deeply
	•	Compute precision, recall, F1-score, accuracy
	•	Understand when each metric matters
	•	Use Scikit-Learn to evaluate your Titanic model
	•	Interpret metric trade-offs
'''

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

titdf = pd.read_csv('./data/week5/titanic_synthetic.csv')

num_cols = ['Age','Fare']
cat_cols = ['Embarked','Sex']
features = num_cols+cat_cols
target = 'Survived'

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scaler',RobustScaler())
])

cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

col_trans = ColumnTransformer([
    ('numeric',num_pipe,num_cols),
    ('category',cat_pipe,cat_cols)
])

X = titdf[features]
Y = titdf[target]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

final_pipe = Pipeline([
    ('prep',col_trans),
    ('model',LogisticRegression(max_iter=1000))
])

final_pipe.fit(X_train,Y_train)
finalYpred = final_pipe.predict(X_test)
acc = accuracy_score(Y_test,finalYpred)
print(acc)

cm = confusion_matrix(Y_test,finalYpred)
print(cm)

clsrprt = classification_report(Y_test,finalYpred)
print(clsrprt)

# print(titdf['Survived'].value_counts(normalize=True))
# print(titdf['Survived'].mean())