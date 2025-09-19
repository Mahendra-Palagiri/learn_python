import pandas as pd
import sklearn.datasets as skd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

iris = skd.load_iris()
fdf = pd.DataFrame(iris.data, columns=iris.feature_names)

trgt = pd.Series(iris.target)

X_train,X_test,Y_train,Y_test = train_test_split(fdf,trgt,test_size=0.2,random_state=42,stratify=trgt)

#Pram grid
param_grid ={
    "n_neighbors" : [3,5,7,9,11,13,15]
}

#Intialize empty model
knn  = KNeighborsClassifier()

#Grid Search with 5 Fold  (cv=5) data sets for cross validation
grdsrchcv = GridSearchCV(knn,param_grid,cv=5,scoring='accuracy')
grdsrchcv.fit(X_train,Y_train)

print("Best Parameters", grdsrchcv.best_params_)
print("Best Score",grdsrchcv.best_score_)

bestknn = grdsrchcv.best_estimator_
Y_pred = bestknn.predict(X_test)
accscore = accuracy_score(Y_test,Y_pred)
print("accscore",accscore)

rparam = {
    "n_neighbors" : range(1,30), #Wider range of n_neighors
    "weights": ['uniform','distance']
}

rknn = KNeighborsClassifier()

# Randomized search (10 random combos, 5-fold CV)
random_search = RandomizedSearchCV(rknn,rparam,n_iter=10,cv=5,scoring='accuracy',random_state=42)

random_search.fit(X_train,Y_train)

print("Best Random params",random_search.best_params_)
print("Best random score",random_search.best_score_)

rbestknn = random_search.best_estimator_
Y_rpred  = rbestknn.predict(X_test)
raccscore = accuracy_score (Y_test,Y_rpred)
print(raccscore)



#Models including Scalar and pipleline)
knnpipeline = Pipeline ([
    ('Scaler',StandardScaler()),
    ('knn',KNeighborsClassifier())
])

plparam_grid = {
    'knn__n_neighbors' : [3,5,7,9,11,13,15]
}

plgrdsrch = GridSearchCV(knnpipeline,plparam_grid,cv=5,scoring='accuracy')
plgrdsrch.fit(X_train,Y_train)

print("Best score with Pipeline and Scaler",plgrdsrch.best_score_)
print("Best params", plgrdsrch.best_params_)