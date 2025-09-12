import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay,roc_auc_score,roc_curve

iris = skd.load_iris()
fdf = pd.DataFrame(iris.data, columns= iris.feature_names)
trgt = pd.Series(iris.target)

X_train,X_test,Y_train,Y_test = train_test_split(fdf,trgt,train_size=0.2,random_state=42,stratify=trgt)

lrpipleline = Pipeline([
    ("sclr",StandardScaler()),
    ("lrmdl",LogisticRegression(max_iter=1000))
])
lrpipleline.fit(X_train,Y_train)

#Binarize labels (convert 3-class labels into one-hot form)
Y_test_bin = label_binarize(Y_test,classes=[0,1,2])
n_classes = Y_test_bin.shape[1]

# Get model probabilities
# ROC requires probabilities, not just predictions
lrp_y_pred = lrpipleline.predict_proba(X_test)  #Proba is available for Logistic regression

#Compute ROC and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i],tpr[i],_ = roc_curve(Y_test_bin[:,i],lrp_y_pred[:,i])
    roc_auc[i] = roc_auc_score(Y_test_bin[:,i],lrp_y_pred[:,i])

plt.figure(figsize=(8,6))
for i, class_name in enumerate(iris.target_names):
    plt.plot(fpr[i], tpr[i], label=f"{class_name} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1], [0,1], "k--")   # diagonal = random guessing
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves (Logistic Regression)")
plt.legend(loc="lower right")
plt.show()


