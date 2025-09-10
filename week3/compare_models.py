import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay

iris = skd.load_iris()

fdf = pd.DataFrame(iris.data, columns=iris.feature_names) #Features

tgt = pd.Series(iris.target) #Targets

X_train,X_test,Y_train,Y_test=train_test_split(fdf,tgt,test_size=0.2,random_state=42,stratify=tgt)


#K Neighbor Classfier
knmodel = KNeighborsClassifier(n_neighbors=5)
knmodel.fit(X_train,Y_train)
knypred = knmodel.predict(X_test)
knacc  = accuracy_score(Y_test,knypred)
# print("\n Accuracy of KN Classifier Model with N size of 5 --> ",knacc)
knreport = classification_report(Y_test,knypred,target_names=iris.target_names)
# print("\n KN Clasification report data  --> \n",knreport)
ConfusionMatrixDisplay.from_estimator(knmodel,X_test,Y_test)
plt.title("KNeighborClassifier")
#plt.show()

#Logisitic Regression
lrmodel = LogisticRegression(max_iter=1000)
lrmodel.fit(X_train,Y_train)
lrypred = lrmodel.predict(X_test)
lracc = accuracy_score(Y_test,lrypred)
# print("\n Accuracy of Logistict Regression Model with max iteration of 1000--> ",lracc)
lrreport = classification_report(Y_test,lrypred,target_names=iris.target_names)
# print("\n LR Clasification report data  --> \n",lrreport)
ConfusionMatrixDisplay.from_estimator(lrmodel,X_test,Y_test)
plt.title("LogistictRegression")
#plt.show()

#Decision Tree
dtmodel = DecisionTreeClassifier(random_state=42)
dtmodel.fit(X_train,Y_train)
dtypred = dtmodel.predict(X_test)
dtaccr = accuracy_score(Y_test,dtypred)
# print("\n Accuracy Score of Decision tree. --> ",dtaccr)
dtreport = classification_report(Y_test,dtypred,target_names=iris.target_names)
# print("\n Decision Tree Report. --> \n",dtreport)
ConfusionMatrixDisplay.from_estimator(dtmodel,X_test,Y_test)
plt.title("DecisionTreeClassifier")
#plt.show()



print("\n\n\n--------------------------------------------------------------------------")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit transform should only be applied on train set to estimate mean, std , median etc
X_test_scaled = scaler.transform(X_test) #transform will use the learnings/outputs from above i.e. from train set and apply on test set

knscaledmodel = KNeighborsClassifier(n_neighbors=5)
knscaledmodel.fit(X_train_scaled,Y_train)
knscaledmodelypred  = knscaledmodel.predict(X_test_scaled)
knscaledmodelacc  = accuracy_score(Y_test,knscaledmodelypred)
knscaledmodelrpt = classification_report(Y_test,knscaledmodelypred,target_names=iris.target_names)

print("\n KN Clasification report data without scaling  --> \n",knreport)
print("\n KN Clasification report data with scaling  --> \n",knscaledmodelrpt)


print("\n\n\n--------------------------------------------------------------------------")
lrscaledmodel = LogisticRegression(max_iter=1000)
lrscaledmodel.fit(X_train_scaled,Y_train)
lrscaledmodelypred  = lrscaledmodel.predict(X_test_scaled)
lrscaledmodelacc  = accuracy_score(Y_test,lrscaledmodelypred)
lrscaledmodelrpt = classification_report(Y_test,lrscaledmodelypred,target_names=iris.target_names)

print("\n LogisticRegression report data without scaling  --> \n",lrreport)
print("\n LogisticRegression report data with scaling  --> \n",lrscaledmodelrpt)


print("\n\n\n--------------------------------------------------------------------------")
dtscaledmodel = DecisionTreeClassifier(random_state=42)
dtscaledmodel.fit(X_train_scaled,Y_train)
dtscaledmodelypred  = dtscaledmodel.predict(X_test_scaled)
dtscaledmodelacc  = accuracy_score(Y_test,dtscaledmodelypred)
dtscaledmodelrpt = classification_report(Y_test,dtscaledmodelypred,target_names=iris.target_names)

print("\n DecisionTreeClassifier report data without scaling  --> \n",dtreport)
print("\n DecisionTreeClassifier report data with scaling  --> \n",dtscaledmodelrpt)

# ----------------- OUTPUT ----------------
# --------------------------------------------------------------------------

#  KN Clasification report data without scaling  --> 
#                precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        10
#   versicolor       1.00      1.00      1.00        10
#    virginica       1.00      1.00      1.00        10

#     accuracy                           1.00        30
#    macro avg       1.00      1.00      1.00        30
# weighted avg       1.00      1.00      1.00        30


#  KN Clasification report data with scaling  --> 
#                precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        10
#   versicolor       0.83      1.00      0.91        10
#    virginica       1.00      0.80      0.89        10

#     accuracy                           0.93        30
#    macro avg       0.94      0.93      0.93        30
# weighted avg       0.94      0.93      0.93        30




# --------------------------------------------------------------------------

#  LogisticRegression report data without scaling  --> 
#                precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        10
#   versicolor       1.00      0.90      0.95        10
#    virginica       0.91      1.00      0.95        10

#     accuracy                           0.97        30
#    macro avg       0.97      0.97      0.97        30
# weighted avg       0.97      0.97      0.97        30


#  LogisticRegression report data with scaling  --> 
#                precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        10
#   versicolor       0.90      0.90      0.90        10
#    virginica       0.90      0.90      0.90        10

#     accuracy                           0.93        30
#    macro avg       0.93      0.93      0.93        30
# weighted avg       0.93      0.93      0.93        30




# --------------------------------------------------------------------------

#  DecisionTreeClassifier report data without scaling  --> 
#                precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        10
#   versicolor       0.90      0.90      0.90        10
#    virginica       0.90      0.90      0.90        10

#     accuracy                           0.93        30
#    macro avg       0.93      0.93      0.93        30
# weighted avg       0.93      0.93      0.93        30


#  DecisionTreeClassifier report data with scaling  --> 
#                precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        10
#   versicolor       0.90      0.90      0.90        10
#    virginica       0.90      0.90      0.90        10

#     accuracy                           0.93        30
#    macro avg       0.93      0.93      0.93        30
# weighted avg       0.93      0.93      0.93        30