# ================================================================
# 📄 Scikit-learn Cheat Sheet — Common Functions & Workflows
# ================================================================

# 1️⃣ Data Loading
from sklearn.datasets import load_iris, load_boston
iris = load_iris()  # iris.data, iris.target

# 2️⃣ Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encoding
encoder = OneHotEncoder()
encoded = encoder.fit_transform(categorical_data)

label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# Missing Values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 4️⃣ Model Selection & Validation
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Hyperparameter Tuning
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

# 5️⃣ Models
# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 6️⃣ Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, mean_squared_error, r2_score
)

# Example:
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7️⃣ Pipelines
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

# ================================================================
# 🧠 Tips:
# - Always split BEFORE scaling to avoid data leakage
# - Use `Pipeline` to keep preprocessing and modeling consistent
# - Use `random_state` for reproducibility
# - Check `help(function)` or `?function` in Jupyter for docs
# ================================================================