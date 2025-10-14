'''
🎯 Objective

Design a complete ML workflow that:
	1.	Loads and explores the Iris dataset
	2.	Scales and preprocesses data
	3.	Trains multiple models
	4.	Tunes hyperparameters using GridSearchCV and RandomizedSearchCV
	5.	Evaluates them on accuracy, F1, and ROC-AUC
	6.	Visualizes feature importance and ROC curves
	7.	Picks the best model and explains why

'''

import sklearn.datasets as skd
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#1. Loads and explores the Iris dataset
iris = skd.load_iris()
featdataframe = pd.DataFrame(iris.data, columns=iris.feature_names)

trgt = pd.Series(iris.target, name='target')
# print(featdataframe)
# print(trgt)


# 2. Scale and Preprocess data
X_train,X_test,Y_train,Y_test = train_test_split(featdataframe,trgt,test_size=0.2,random_state=42,stratify=trgt)

knnpipeline = Pipeline([
    ("Scaler",StandardScaler()),
    ("knn",KNeighborsClassifier(n_neighbors=5))
])

lgregrpipeline = Pipeline([
    ("Scaler",StandardScaler()),
    ("lgregr",LogisticRegression(max_iter=1000))
])


#3. Train multiple models
knnpipeline.fit(X_train,Y_train)
knnpred = knnpipeline.predict(X_test)
knnreport = classification_report(Y_test,knnpred,target_names=iris.target_names)
# print("\n KNN report with standard scaler --> \n",knnreport)

lgregrpipeline.fit(X_train,Y_train)
lgregrpred = lgregrpipeline.predict(X_test)
lgregrreport = classification_report(Y_test,lgregrpred,target_names=iris.target_names)
# print("\n Linear regression report with standard scaler --> \n",lgregrreport)

dtclfr = DecisionTreeClassifier(max_depth=3, random_state=42)
dtclfr.fit(X_train,Y_train)
dtclfrpred = dtclfr.predict(X_test)
dtclfrreport = classification_report(Y_test,dtclfrpred,target_names=iris.target_names)
# print("\n Decision report with standard scaler --> \n",dtclfrreport)



#4. Tunes hyperparameters using GridSearchCV and RandomizedSearchCV
#KNN
knnpln = Pipeline([
    ("Scaler",StandardScaler()),
    ("knn",KNeighborsClassifier())
])

knnparams = {
    "knn__n_neighbors" : [1,3,5,7,9,13,15,17,19]
}

knngrdsearch = GridSearchCV(knnpln,knnparams,cv=5,scoring="accuracy")
knngrdsearch.fit(X_train,Y_train)
# print("\n Grid Serach CV using KNN  (Best Params)--> ",knngrdsearch.best_params_)
# print("\n Grid Serach CV using KNN  (Best Score)--> ",knngrdsearch.best_score_)

knnbest = knngrdsearch.best_estimator_
knnbestpred = knnbest.predict(X_test)
knnbestreport = classification_report(Y_test,knnbestpred,target_names=iris.target_names)
print("\n KNN grid search CV report --> \n",knnbestreport)

#Random Search
knnrndserachparams = {
    "knn__n_neighbors" : range(1,30)
}

knnrndsrch = RandomizedSearchCV(knnpln,knnrndserachparams,n_iter=10,cv=5,scoring="accuracy")
knnrndsrch.fit(X_train,Y_train)
# print("\n Random search CV using KNN  (Best Params)--> ",knnrndsrch.best_params_)
# print("\n Random search CV using KNN  (Best Score)--> ",knnrndsrch.best_score_)

knnrndmbest = knnrndsrch.best_estimator_
knnrndmbestpred = knnrndmbest.predict(X_test)
knnrndmbestrprt = classification_report(Y_test,knnrndmbestpred,target_names=iris.target_names)
print("\n KNN random search CV report --> \n",knnrndmbestrprt)

#LogisticRegression
lrgrppline = Pipeline([
     ("Scaler",StandardScaler()),
     ("lrgr",LogisticRegression())
])

lrgtparams = {
    "lrgr__max_iter" :[100,200,300,1000,2000]
}

lrgrsearch = GridSearchCV(lrgrppline,lrgtparams,cv=5,scoring="accuracy")
lrgrsearch.fit(X_train,Y_train)
# print("\n Grid Serach CV using LogisticRegression  (Best Params)--> ",lrgrsearch.best_params_)
# print("\n Grid Serach CV using LogisticRegression  (Best Score)--> ",lrgrsearch.best_score_)

lrgrbestestimator = lrgrsearch.best_estimator_
lrgrbestestimatorpred = lrgrbestestimator.predict(X_test)
lrgrbestestimatorrpt = classification_report(Y_test,lrgrbestestimatorpred,target_names=iris.target_names)
print("\n KNN grid search CV report --> \n",lrgrbestestimatorrpt)


lrgrndmparams = {
   "lrgr__max_iter" : range(100,2000,100)
}
lrgrrandsearch = RandomizedSearchCV(lrgrppline,lrgrndmparams,n_iter=10,cv=5,scoring="accuracy")
lrgrrandsearch.fit(X_train,Y_train)
# print("\n Random search CV using Logistic Regression  (Best Params)--> ",lrgrrandsearch.best_params_)
# print("\n Random search CV using Logistic Regression  (Best Score)--> ",lrgrrandsearch.best_score_)

lrgrrandbestestimator = lrgrrandsearch.best_estimator_
lrgrrandbestestimatorpred = lrgrrandbestestimator.predict(X_test)
lrgrrandbestestimatorrpt = classification_report(Y_test,lrgrrandbestestimatorpred,target_names=iris.target_names)
print("\n LR random search CV report --> \n",lrgrrandbestestimatorrpt)

#Decision Tree
dtparams = {
    "max_depth": [1,2,3,4,5,6,7,8,9]
}

dtrndparams = {
     "max_depth": range(1,20)
}

dectree = DecisionTreeClassifier()

dtgrdsearch = GridSearchCV(dectree,dtparams,cv=5,scoring="accuracy")
dtrndsearch = RandomizedSearchCV(dectree,dtrndparams,n_iter=10,cv=5,scoring="accuracy")

dtgrdsearch.fit(X_train,Y_train)
dtrndsearch.fit(X_train,Y_train)

dtgrdsrchbestestimator = dtgrdsearch.best_estimator_
dtrndsrchbestestimator = dtrndsearch.best_estimator_

print("\nClassification report for GS on DT \n",classification_report(Y_test,dtgrdsrchbestestimator.predict(X_test),target_names=iris.target_names))
print("\nClassification report for RS on DT \n",classification_report(Y_test,dtrndsrchbestestimator.predict(X_test),target_names=iris.target_names))