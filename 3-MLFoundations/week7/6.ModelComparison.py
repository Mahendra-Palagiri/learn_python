'''Week 7 — Day 6: Model Comparison (Fair Comparison Using the Same CV Protocol)

📚 Topics
	•	Comparing multiple model families fairly (same data, same folds, same metric)
	•	Reusing the same preprocessing pipeline for all models
	•	Interpreting results: CV mean vs std (performance vs stability)
	•	Why a single Decision Tree can overfit, and why Random Forest reduces that risk
	•	Making a defensible “winner pick” based on evidence

🎯 Learning Goals
	•	Run 5-fold Stratified CV for at least 3 models on our synthetic Titanic dataset
	•	Compare each model’s mean F1 and std
	•	Pick the best model and explain the choice in 2 lines (mean + stability)

'''


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#1. Import the titanic data set
titdf = pd.read_csv('./data/week5/titanic_synthetic.csv')
# print(titdf.info())
# print(titdf.describe(include='all'))

#2. Categorize Columns and split the dataframe
num_cols =['Age','Fare']
cat_cols=['Sex','Embarked']
target ='Survived'
features = num_cols +cat_cols
X = titdf[features]
Y = titdf[target]

#3. Create basic pipelines
num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scale',RobustScaler())
])

cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(handle_unknown='ignore'))
])

#4. Create Pre-Processor and Final Pipleline
pre_prcsr = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

models ={
    "LogReg": LogisticRegression(max_iter=1000,solver='liblinear'),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest" : RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
}

for name,model in models.items():
    pipe = Pipeline([('preprcsr',pre_prcsr),('model',model)])
    scores = cross_val_score(estimator=pipe,X=X,y=Y,cv=cv,scoring='f1')
    print(f"{name:12s} | mean={scores.mean():.6f}  std={scores.std():.6f}  folds={np.round(scores, 6)}")


'''
LogReg       | mean=0.649511  std=0.013166  folds=[0.663016 0.638362 0.667996 0.639839 0.63834 ]
DecisionTree | mean=0.553602  std=0.006894  folds=[0.556874 0.565497 0.547959 0.546813 0.550868]
RandomForest | mean=0.591260  std=0.006972  folds=[0.596439 0.579739 0.598606 0.586946 0.594568]


1) What the numbers say
	•	LogReg: mean 0.6495, std 0.0132
	•	DecisionTree: mean 0.5536, std 0.0069
	•	RandomForest: mean 0.5913, std 0.0070

So on this dataset + feature set:
	•	LogisticRegression is the clear winner on mean F1.
	•	Tree and Forest are more stable across folds (lower std), but their mean performance is notably lower.

2) Winner and why (2 lines)
	•	Winner: LogReg — highest mean F1 (0.6495), best overall performance.
	•	Std is higher than Tree/Forest, but still fairly small; stability is acceptable.

3) What this implies (quick diagnosis)

This looks like the data/features are more “linear-friendly” 
(or the synthetic generation makes it so). 
The non-linear models aren’t finding extra signal; instead they’re underperforming.

Also: the train-vs-test CV gap from Day 5 was basically zero, 
so LogReg isn’t screaming “overfitting.” That supports keeping it as the final choice.

'''