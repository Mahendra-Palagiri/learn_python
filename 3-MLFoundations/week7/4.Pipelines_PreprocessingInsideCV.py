'''Week 7 — Day 4: Pipelines + Preprocessing Inside CV (Leakage-Proof Workflow)

📚 Topics
	•	Why preprocessing must happen inside cross-validation folds
	•	Pipeline (chain steps: preprocess → model)
	•	ColumnTransformer (different preprocessing for numeric vs categorical)
	•	Correct evaluation with cross_val_score / GridSearchCV on the pipeline
	•	The “smell test”: how to recognize leakage in code

🎯 Learning Goals
	•	Explain (in one sentence) why preprocessing outside CV is leakage
	•	Build a Pipeline that includes preprocessing + model
	•	Use CV on the Pipeline and interpret fold scores (mean + std)
	•	Describe the “correct pattern” that will be reused in Day 5 tuning

'''

# ==========================================================
# 1 — The core rule
# ==========================================================
'''
    Anything that learns from data (imputer, scaler, encoder, feature selection, PCA, etc.) 
    must be fit only on training data — and in CV that means fit inside each fold.

    That’s exactly what Pipeline guarantees.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression

#1 Load titanic Dataset
titdf = pd.read_csv("./data/week5/titanic_synthetic.csv")
# print(titdf.info())
# print(titdf.describe(include='all'))

#2 Define features and target
target = 'Survived'
num_cols = ['Age','Fare']
cat_cols = ['Sex','Embarked']
features = num_cols + cat_cols

X = titdf[features]
Y = titdf[target]

#3 Preprocessing pipelines
num_pipe = Pipeline([
    ("impute",SimpleImputer(strategy='median')),
    ("scale",RobustScaler())
])

cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

pre_proces = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])

#4 Final Pipleline
final_pipe = Pipeline([
    ('preprcsr',pre_proces),
    ('model',LogisticRegression(random_state=42,max_iter=1000))
])

#5 CV (Stratified for classification)
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

scores = cross_val_score(final_pipe,X, Y, cv=cv,scoring='f1')

print("Fold F1 Scores :",scores)
print("F1 mean : ",scores.mean())
print("F1 std: ",scores.std())

'''
    Fold F1 Scores : [0.66301643 0.63836164 0.66799601 0.63983903 0.63833992]
    F1 mean :  0.6495106063124444
    F1 std:  0.013166171764503045
'''


#=========================================================
# Summarization in simple words
#=========================================================
'''
    This is kind of repeat of what we did in Week 5

    We used titanic dataframe for our use case
    We identified numeric columns, category columns  (combinedly called as features) and target

    we split the titain datframe with features and target

    We then constructed numeric pipeline with robustscalear and simpleimputer 
    we constructed category pipeline with simpleimpurter and onehotencoder

    we combined both the above pipleine using columntransformer and called it as preprocessing
    Finally we used the preprocessing and Logistricregression model to construct a final pipleine

    we defined CV using StratifiedKFold with K=5 and shuffle = true

    then we finally got the cross validation scroe using the final pipeline, feature dataset, target dataset and scored on F1

    The following was our output
    Fold F1 Scores : [0.66301643 0.63836164 0.66799601 0.63983903 0.63833992]
    F1 mean :  0.6495106063124444
    F1 std:  0.013166171764503045

    Given the f1 std is 0.013 which is low means that the model is stable across various CV folds
    Given the F1 mean is 0.64... this is a good performance and there is chance to improve this further. (which we will take up in next coming days)

'''