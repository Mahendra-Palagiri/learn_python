'''
    🌅 Week 4 – Day 3: Building a Full Preprocessing + Model Pipeline

    🎯 Learning Goals

    By the end of today we’ll:
        1.	Combine multiple preprocessing steps (imputation + scaling + encoding) in one reproducible pipeline.
        2.	Train and evaluate a simple Logistic Regression model safely (no data leakage).
        3.	Understand how pipelines keep preprocessing consistent during both training and prediction.
        4.	Save and reload your trained pipeline using joblib.
'''


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay,confusion_matrix
import joblib

# 1. Import Data
datafr = pd.read_csv("./data/week4/titanic_with_sex.csv")
# print("\n\n",datafr.info())
# print("\n\n",datafr.head(3))
# print("\n\n",datafr.describe(include='all'))

# 2. Prepare Data
features = ['Age','Fare','Embarked','Sex']
target  ='Survived'

X = datafr[features]
Y = datafr[target]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# 3. Preparing Preprocessors
num_features = ['Age','Fare']
cat_features = ['Embarked','Sex']

num_pipeline = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scale',RobustScaler()) #Robust scaler to ensure that any outliers doesnt skew output
])

cat_pipeline = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('scaler',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num',num_pipeline,num_features),
    ('cat',cat_pipeline,cat_features)
])

# 4. Fitting all together
final_pipeline  = Pipeline([
    ('prep',preprocessor),
    ('model',LogisticRegression(max_iter=1000, random_state=42))
])
final_pipeline.fit(X_train,Y_train)

# 5. Evaluate the model
Y_pred = final_pipeline.predict(X_test)
print("\n\n -- ACCURACY SCORE --\n", accuracy_score(Y_test,Y_pred))
print("\n\n -- CLASSIFICATION REPORT --\n", classification_report(Y_test,Y_pred))
print("\n\n -- CONFUSION MATRIX --\n", confusion_matrix(Y_test,Y_pred))

'''        OUTPUT  

    -- ACCURACY SCORE --
    0.75


    -- CLASSIFICATION REPORT --
                precision    recall  f1-score   support

            0       0.86      0.75      0.80         8
            1       0.60      0.75      0.67         4

        accuracy                           0.75        12
    macro avg       0.73      0.75      0.73        12
    weighted avg       0.77      0.75      0.76        12



    -- CONFUSION MATRIX --
    [[6 2]
    [1 3]]

'''

# 6. Store the learnings (Impute, Scale, Modelling) into a file that can be used to reinitialize model for later use
joblib.dump(final_pipeline,"./data/week4/titanic.pk1")

loaded_model = joblib.load("./data/week4/titanic.pk1") #Re-initiazling the model and its learning

# Predict new passengers
new_passengers = pd.DataFrame({
    "Age": [22, 35],
    "Fare": [7.25, 53.1],
    "Embarked": ["S", "C"],
    "Sex": ["male", "female"]
})

print("\n--- MODEL PREDICTION FOR NEW DATA\n\n",loaded_model.predict(new_passengers))

# Checking which pramaters influenced the model
features = final_pipeline.named_steps['prep'].get_feature_names_out()
weights = final_pipeline.named_steps['model'].coef_[0]
xdf = pd.DataFrame({'Feature': features, 'Weight': weights}).sort_values('Weight', ascending=False)
print(xdf)


''' --- MINI EXCERCISE -----

    1. Print the transformed feature names
    2. Try switching the model from LogisticRegression → DecisionTreeClassifier()
        and check if scaling makes any difference to accuracy
    3. Inspect one encoded row
'''

# ME1) Print the transformed feature names
featurenames = final_pipeline.named_steps["prep"].get_feature_names_out()
print("\n\n --- TRANSFORMED FEATURE NAMES ---- \n\n",featurenames)

# ME2) Try switching the model from LogisticRegression → DecisionTreeClassifier()
# and check if scaling makes any difference to accuracy


# We cannot use decision tree on string (unless we encode so using string encoding)
xprocessor = ColumnTransformer([('cat',cat_pipeline,cat_features)])
dtunscaledpipe = Pipeline([
    ('prep',xprocessor),
    ('mdl',DecisionTreeClassifier(max_depth=3, random_state=42))
])
dtunscaledpipe.fit(X_train,Y_train)
unsclaedypred = dtunscaledpipe.predict(X_test)
unscaledacc = accuracy_score(Y_test,unsclaedypred)
unscaledclassrprt = classification_report(Y_test,unsclaedypred)

dtpipeline = Pipeline([
    ('prep',preprocessor),
    ('model',DecisionTreeClassifier(max_depth=3,random_state=42))
])
dtpipeline.fit(X_train,Y_train)
dtscaledypred = dtpipeline.predict(X_test)
dtscaledacc = accuracy_score(Y_test,dtscaledypred)
dtclassrprt = classification_report(Y_test,dtscaledypred)

comapredf = pd.DataFrame({
    "modeltype": ['Unscaled','Scaled'],
    "AccuracyScore": [unscaledacc,dtscaledacc],
    "ClasificationReport": [unscaledclassrprt,dtclassrprt]
})
print(comapredf)



'''  ------------ RETROSPECTION --------------

    🧩 Confusion Matrix Structure

    For a binary classification problem like Survived (0 = Did not survive, 1 = Survived),
    a confusion matrix compares actual (true) vs predicted labels.

    It’s always structured as:

                    Predicted = 0           Predicted = 1
    ----------      --------------          ----------------                
    Actual = 0      True Negative (TN)      False Positive (FP)
    Actual = 1      False Negative (FN)     True Positive (TP)

    our confusion matrix 
     -- CONFUSION MATRIX --
    [[6 2]
    [1 3]]

                        Predicted = 0           Predicted = 1
    ----------      --------------          ----------------                
    Actual = 0      6                       2
    Actual = 1      1                       3

    🧠 Interpreting Each Number
    Term    Meaning                                                 Value   What it means here
    ----    ------------------------------------                    -----   -------------------------
    TN      Model predicted did not survive (0) correctly           6       6 passengers correctly predicted as not surviving
    FP      Model predicted survived (1) but actually didn’t        2       2 passengers wrongly predicted as survivors
    FN      Model predicted did not survive (0) but actually did    1       1 actual survivor missed by the model
    TP      Model predicted survived (1) correctly                  3       3 survivors correctly predicted

    Derived Metrics

    You can compute your main evaluation scores directly from these four values:
    Metric                      Formula                             Substitution                Result  
    ----------                  ---------                           -------------               ---------    
    Accuracy                    (TP + TN) / Total                   (3 + 6) / 12                0.75 (75%)
    Precision (for class 1)     TP / (TP + FP)                      3 / (3 + 2)                 0.6 (60%)
    Recall (for class 1)        TP / (TP + FN)                      3 / (3 + 1)                 0.75 (75%)
    F1-Score                    2 × (Prec × Rec) / (Prec + Rec)     2×(0.6×0.75)/(0.6+0.75)     ≈ 0.67 (67%
    


'''