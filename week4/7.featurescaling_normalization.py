import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,RobustScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def accr_coeff_with_diff_scaling(scaler):
    
    numcols = ['Age','Fare']
    catcols = ['Embarked','Sex']
    features = numcols + catcols
    target = 'Survived'

    num_pipe = Pipeline([
        ('impute',SimpleImputer(strategy='median')),
        ('scale',scaler)
    ])

    cat_pipe = Pipeline([
        ('impute',SimpleImputer(strategy='most_frequent')),
        ('encode',OneHotEncoder(handle_unknown='ignore'))
    ])

    preprcs = ColumnTransformer([
        ('num',num_pipe,numcols),
        ('catg',cat_pipe,catcols)
    ])

    final_pipe = Pipeline([
        ('preprcs',preprcs),
        ('model',LogisticRegression(max_iter=1000))
    ])
    
    X = titdf[features]
    Y = titdf[target]

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    final_pipe.fit(X_train,Y_train)
    finalypred = final_pipe.predict(X_test)
    acc = accuracy_score(Y_test, finalypred)

    ohe = final_pipe.named_steps['preprcs'].named_transformers_['catg'].named_steps['encode']
    cat_feature_names = ohe.get_feature_names_out(catcols)
    all_feature_names = numcols + list(cat_feature_names)

    coeffs = final_pipe.named_steps['model'].coef_[0]
    coeff_df = pd.DataFrame({
        'feature': all_feature_names,
        'coef': coeffs
    }).sort_values(by='coef', key=abs, ascending=False)

    return acc, coeff_df


titdf = pd.read_csv('./data/week4/titanic_with_sex.csv')

acc_std, coef_std = accr_coeff_with_diff_scaling(StandardScaler())
acc_rb,  coef_rb  = accr_coeff_with_diff_scaling(RobustScaler())

print(acc_std, acc_rb)
print(coef_std.head(10))
print(coef_rb.head(10))

