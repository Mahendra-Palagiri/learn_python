# Week 3 - Day 4
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skd
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


iris  = skd.load_iris()
fdf = pd.DataFrame(iris.data, columns=iris.feature_names)

trgts = pd.Series(iris.target)

#Startify ensures the data set split for train and test is uniform acrosss targets (if we dont specify then it might lead to unbalanced target size skweing up the model output)
X_train,X_test,Y_train,Y_test = train_test_split(fdf,trgts,test_size=0.2,random_state=42,stratify=trgts) 


#KNeighborsClassifier with Pipeline
pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("knn",KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(X_train,Y_train)

pypred = pipeline.predict(X_test)

paccscore = accuracy_score(Y_test,pypred)

preport = classification_report(Y_test,pypred,target_names=iris.target_names)

print("Accuracy  --> ",paccscore)
print("Report data --> \n",preport)

ConfusionMatrixDisplay.from_estimator(pipeline,X_test,Y_test)
plt.title("KNNPipleline")
# plt.show()


#LogisticRegression with pipleine

lrpipeline  = Pipeline([
        ("scaler",StandardScaler()),
        ("logregr",LogisticRegression(max_iter=1000))
])
lrpipeline.fit(X_train,Y_train)
lrpypred = lrpipeline.predict(X_test)
lrpaccscore = accuracy_score(Y_test,lrpypred)
lrpreport = classification_report(Y_test,lrpypred,target_names=iris.target_names)
print("LR Accuracy  --> ",lrpaccscore)
print("LR Report data --> \n",lrpreport)

ConfusionMatrixDisplay.from_estimator(lrpipeline,X_test,Y_test)
plt.title("LRPipleline")
# plt.show()

'''
🔹 Feature Importance
	•	Some models tell us which features matter most in prediction.
	•	Decision Tree / Random Forest → model.feature_importances_.
	•	Logistic Regression → coefficients (model.coef_).
	•	KNN → no native feature importance (distance-based).
'''

#Get coefficients
coefficients  = lrpipeline.named_steps["logregr"].coef_

feature_importance_lr = pd.DataFrame(coefficients.T,index=iris.feature_names,columns=iris.target_names)
print("\n Logistic regression feature Importance -->\n",feature_importance_lr)



#Decision Tree 
dt_pipeline = Pipeline([
    ("dt",DecisionTreeClassifier(random_state=42))
])
dt_pipeline.fit(X_train,Y_train)
dtypred = dt_pipeline.predict(X_test)
dtaccscore = accuracy_score (Y_test,dtypred)
dtreport = classification_report(Y_test,dtypred,target_names=iris.target_names)
print("DT Accuracy  --> ",dtaccscore)
print("DT Report data --> \n",dtreport)

ConfusionMatrixDisplay.from_estimator(dt_pipeline,X_test,Y_test)
plt.title("DecisionTreeClassifier")
# plt.show()

feature_impt = dt_pipeline.named_steps["dt"].feature_importances_

feature_importance_dt = pd.DataFrame({
    "Feature": iris.feature_names,
    "Importance": feature_impt
}).sort_values(by="Importance", ascending=False)

print("\n Decision tree feature Importance -->\n",feature_importance_dt)

'''
1. How do we know that we have to use coef_ for logistic regression and feature_names_ for decision tree 
2. Why is the structure of dataframe consturcted differnetly for LogisticRegression vs DecisionTree

🔹 Why coef_ for Logistic Regression?
	•	Logistic Regression is a linear model.
	•	Its decision rule is essentially:

score_c = (w_1 x_1 + w_2 x_2 + … + w_n x_n + b)
	•	The weights (w) are stored in .coef_.
	•	Shape of .coef_:
	•	For binary classification → (1, n_features)
	•	For multi-class classification (Iris has 3 classes) → (n_classes, n_features)

👉 That’s why we transpose it (.T) to align features as rows and classes as columns when making a DataFrame.

⸻

🔹 Why feature_importances_ for Decision Tree?
	•	Decision Trees work by splitting data at thresholds (petal length ≤ 2.45).
	•	Each split reduces impurity (gini or entropy).
	•	The model tracks how much each feature contributes to reducing impurity.
	•	This info is stored in .feature_importances_.
	•	Shape of .feature_importances_: (n_features,) → just one importance score per feature.

👉 That’s why the DataFrame is constructed as one column: Feature vs Importance.

⸻

🔹 Why the DataFrames are different
	•	Logistic Regression → multi-class → each class gets its own weight per feature.
	•	DataFrame needs feature names as rows, class labels as columns.
	•	Decision Tree → one global measure per feature.
	•	DataFrame just has two columns: Feature, Importance.

⸻

✅ Quick Visual

Logistic Regression (coef_)
            setosa  versicolor  virginica
sepal len   0.12     -0.45       0.33
sepal wid   0.02      0.10      -0.12
...

Decision Tree (feature_importances_)
    Feature             Importance
0   petal length (cm)   0.58
1   petal width (cm)    0.30
2   sepal length (cm)   0.09
3   sepal width (cm)    0.03
'''