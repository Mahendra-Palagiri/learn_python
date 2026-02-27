import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve
import matplotlib.pyplot as plt 

titdf = pd.read_csv('./data/week5/titanic_synthetic.csv')

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

pre_prcsr = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('catg',cat_pipe,cat_cols)
])

X = titdf[features]
Y = titdf[target]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


#Model A. --> Baseline model (L2)
l2_model = LogisticRegression(
    penalty='l2',
    C=1,
    solver='lbfgs',
    max_iter=1000
)

l2_pipe = Pipeline([
    ('preprcsr',pre_prcsr),
    ('l2model',l2_model)
])

l2_pipe.fit(X_train,Y_train)
l2Y_predict = l2_pipe.predict(X_test)
l2Y_proba = l2_pipe.predict_proba(X_test)[:,1]

l2_metrics = {
    "Accuracy" : accuracy_score(Y_test,l2Y_predict),
    "Precision" : precision_score(Y_test,l2Y_predict),
    "Recall" : recall_score(Y_test,l2Y_predict),
    "F1" : f1_score(Y_test,l2Y_predict),
    "ROC-AUC": roc_auc_score(Y_test,l2Y_proba)
}



#Model B. --> Tuned model (best estimator)
params_grid = {
    'model__C':[0.01,0.1,1,10],
    'model__penalty': ['l1','l2'],
    'model__solver': ['liblinear']
}

final_pipe = Pipeline([
    ('preprcsr',pre_prcsr),
    ('model',LogisticRegression(max_iter=1000))
])

grid = GridSearchCV(
    estimator=final_pipe,
    param_grid=params_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)
grid.fit(X_train,Y_train)

best_estimator = grid.best_estimator_
bestY_predict = best_estimator.predict(X_test)
bestY_proba = best_estimator.predict_proba(X_test)[:,1]

best_metrics = {
    "Accuracy" : accuracy_score(Y_test,bestY_predict),
    "Precision" : precision_score(Y_test,bestY_predict),
    "Recall" : recall_score(Y_test,bestY_predict),
    "F1" : f1_score(Y_test,bestY_predict),
    "ROC-AUC": roc_auc_score(Y_test,bestY_proba)
}

#Model C. --> Sparse model (L1)
l1_model = LogisticRegression(
    penalty='l1',
    C=1,
    solver='liblinear',
    max_iter=1000
)

l1_pipe = Pipeline([
    ('preprcsr',pre_prcsr),
    ('l1model',l1_model)
])

l1_pipe.fit(X_train,Y_train)
l1Y_predict = l1_pipe.predict(X_test)
l1Y_proba = l1_pipe.predict_proba(X_test)[:,1]

l1_metrics = {
    "Accuracy" : accuracy_score(Y_test,l1Y_predict),
    "Precision" : precision_score(Y_test,l1Y_predict),
    "Recall" : recall_score(Y_test,l1Y_predict),
    "F1" : f1_score(Y_test,l1Y_predict),
    "ROC-AUC": roc_auc_score(Y_test,l1Y_proba)
}

#Final Dataframe
final_df = pd.DataFrame([
    {"Model": "Baseline (L2)", **l2_metrics},
    {"Model": "Tuned (CV)", **best_metrics},
    {"Model": "Sparse (L1)", **l1_metrics},
])

metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
final_df = final_df[["Model", *metric_cols]].round(4)
print('\n L2 Metrics  --> \n',l2_metrics)
print('\n Best Metrics  --> \n',best_metrics)
print('\n L1 Metrics  --> \n',l1_metrics)
print('\n\n',final_df)


# Working with ROC Curve
models = {
    "Baseline (L2, C=1.0)": l2_pipe,
    "Tuned (GridSearchCV)": best_estimator,
    "Sparse (L1, C=0.1)": l1_pipe
}

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, y_proba)
    plt.plot(fpr, tpr, label=name)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()