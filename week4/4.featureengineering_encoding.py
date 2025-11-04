import pandas as pd
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


titdf  = pd.read_csv("./data/week4/titanic_with_sex.csv")
# print(titdf.info())
# print(titdf.describe(include='all'))

features = ['Age','Fare','Embarked','Sex']
target = 'Survived'

X  = titdf[features]
Y  = titdf[target]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

num_cols = ['Age','Fare']
cat_cols = ['Embarked','Sex']

num_pipe  = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scale',RobustScaler())
])

cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(handle_unknown='ignore'))
])

preprcs = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('catg',cat_pipe,cat_cols)
])

final_pipe = Pipeline([
    ('preprc',preprcs),
    ('model',LogisticRegression(max_iter=1000))
])
final_pipe.fit(X_train,Y_train)
finalypred = final_pipe.predict(X_test)

# print('\n\n ==== ACCURACY SCORE ==\n',accuracy_score(Y_test,finalypred))
# print('\n\n ==== CONFUSION MATRIX ==\n',confusion_matrix(Y_test,finalypred))


# MINI CHALLANGE
# Comparing encoded vs unencoded model accuracy

X_UE = titdf[num_cols]  # Considering only numeric columns for unencoded one (as for strings we cannot use them without encoding)
Y_UE = titdf['Survived']
XUE_train,XUE_test,YUE_train,YUE_test  = train_test_split(X_UE,Y_UE,test_size=0.2,random_state=42)

UE_preprcs = ColumnTransformer([
    ('num',num_pipe,num_cols)
])

UE_final_pipe = Pipeline([
    ('preprc',UE_preprcs),
    ('model',LogisticRegression(max_iter=1000))
])
UE_final_pipe.fit(XUE_train,YUE_train)
UE_finalypred = UE_final_pipe.predict(XUE_test)

# print('\n\n ==== ACCURACY SCORE ==\n',accuracy_score(YUE_test,UE_finalypred))
# print('\n\n ==== CONFUSION MATRIX ==\n',confusion_matrix(YUE_test,UE_finalypred))


print('\n\n ==== ACCURACY SCORE BEFORE AND AFTER ENCODING == \t',accuracy_score(YUE_test,UE_finalypred),'\t',accuracy_score(Y_test,finalypred))




'''. ----------------- RETROSPECTION -----------------------
    🧩 Categorical Columns in the Titanic Dataset

    Your dataset now contains both numeric and categorical features:
    Column      Type                            Example             Comment    
    -------     ------------                    -------             ------------
    Age         Numeric (float)                 32.8                Continuous feature
    Fare        Numeric (float)                 56.8                Continuous feature
    Embarked    Categorical                     “C”, “S”, “Q”       Nominal (no natural order)
    Sex         Categorical                     “male”, “female”    Nominal (binary)
    Cabin       Categorical (many missing)      “C57”, “E34”        High-cardinality, often dropped
    Survived    Target                          0 or 1              Binary classification label

    🧠 Why Encoding Is Necessary

        Machine-learning models work with numbers only.
        If you pass raw strings like "male" or "S", the model can’t compute distances or weights.
        Hence, we must encode categorical values into numeric form without distorting meaning.


    ⚙️ Encoding Options
    Encoder                     Description                                                   Best for                Example Output
    ----------                  ----------------                                              ---------               -------------------
    One-Hot Encoding            Creates new binary columns (1 if category present, else 0)    Nominal data            Embarked_C, Embarked_S, Embarked_Q
    Ordinal Encoding            Replaces each category with an integer label                  Ordinal data (ordered)  CabinLevel A→1, B→2
    Target Encoding (advanced)  Encodes categories using mean of target variable              High-cardinality data    Cabin=A → 0.72 (survival rate)

'''