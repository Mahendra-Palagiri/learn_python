import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score,classification_report

titanidf  = pd.read_csv('./data/week4/titanic_with_sex.csv')
# print(titanidf.info())
# print(titanidf.describe(include='all'))

features = ['Age','Fare','Embarked','Sex']
target ='Survived'

num_cols = ['Age','Fare']
cat_cols = ['Embarked','Sex']

X = titanidf[features]
Y = titanidf[target]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scale',RobustScaler())
])

cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(handle_unknown='ignore'))
])

pre_prcsr = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('catg',cat_pipe,cat_cols)
])

final_pipe = Pipeline([
    ('prepcsr',pre_prcsr),
    ('model',LogisticRegression(max_iter=1000))
])
final_pipe.fit(X_train,Y_train)
finalypred = final_pipe.predict(X_test)
print('\n\n. == LR - ACCURACY SCORE ==>\t',accuracy_score(Y_test,finalypred))
print('\n\n. == LR - CLASSIFICATION REPORT ==>\n',classification_report(Y_test,finalypred))

# print(final_pipe.named_steps['prepcsr'].get_feature_names_out())
# print(final_pipe[:-1].get_feature_names_out()) #--> Other way of getting the feature names out
# print(final_pipe.named_steps['model'].coef_)

print("\n\n",pd.DataFrame({
    'feature_names': final_pipe.named_steps['prepcsr'].get_feature_names_out(),
    'LR_weights': final_pipe.named_steps['model'].coef_[0]
    }).sort_values(['LR_weights'],ascending=False))

dt_final_pipe = Pipeline([
    ('prepcsr',pre_prcsr),
    ('model',DecisionTreeClassifier(max_depth=3))
])
dt_final_pipe.fit(X_train,Y_train)
dtypred = dt_final_pipe.predict(X_test)
print('\n\n. == DT - ACCURACY SCORE ==>\t',accuracy_score(Y_test,dtypred))
print('\n\n. == DT - CLASSIFICATION REPORT ==>\n',classification_report(Y_test,dtypred))

# print(dt_final_pipe[:-1].get_feature_names_out())
# print(dt_final_pipe.named_steps['model'].feature_importances_)

print("\n\n",pd.DataFrame({
    'feature_names': dt_final_pipe.named_steps['prepcsr'].get_feature_names_out(),
    'DT_weights': dt_final_pipe.named_steps['model'].feature_importances_
    }).sort_values(['DT_weights'],ascending=False))


# Select only those features which can influence the outcome
auto_pipe = Pipeline([
    ('prepcsr', pre_prcsr),
    ('select', SelectFromModel(LogisticRegression(max_iter=1000), threshold='median')),
    ('model', LogisticRegression(max_iter=1000))
])

auto_pipe.fit(X_train, Y_train)
y_pred = auto_pipe.predict(X_test)
print("Auto-pipeline accuracy:", accuracy_score(Y_test, y_pred))