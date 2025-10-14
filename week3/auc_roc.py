#Week 3 - Day 5
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


'''
🔹 Why Accuracy Isn’t Always Enough
	•	Accuracy = correct predictions/total predictions.
	•	Works fine for balanced datasets like Iris.
	•	But in imbalanced data (e.g., 95% healthy patients, 5% sick patients):
	•	A model that predicts “healthy” for everyone gets 95% accuracy — but is useless.

🔹  Precision, Recall, F1-Score

Let’s break them down :
	•	Precision → “Of all flowers I said were Virginica, how many really were Virginica?”
         - Formula: 
                Precision} = TP/(TP + FP) --> TP (True Prediction), FP (False Prediction)
         - Translation: How careful the model is when it says Yes.

	•	Recall → “Of all the Virginica flowers, how many did I actually catch?”
	    - Formula: 
            Recall = TP/(TP + FN). --> FN (False Negatives)
	    - Translation: How good the model is at catching everything.

	•	F1-Score → Balance between precision and recall.
	    - Formula:
            F1 = 2 * (Precision × Recall)/(Precision + Recall)
	    - Translation: The teacher’s “overall grade” when you need both neatness and speed.

    Precision vs Recall with an exmaple      

    🔹 Imagine a Medical Test Example (Simple Case)

    Let’s say we’re testing 100 people for a disease:
        •	10 people are actually sick.
        •	90 people are actually healthy.

    Case 1: Precision
    -----------------
    👉 Question: “When the test says someone is sick, how often is it right?”
        •	If the test says 20 people are sick…
        •	8 are actually sick (true positives)
        •	12 are actually healthy (false positives)

    Precision= TP/ (TP + FP)  = 8/(8+12) = 0.40

    ➡️ Only 40% of the positive results were correct → the test gives too many false alarms.

    
    Case 2: Recall
    --------------
    👉 Question: “Of all the sick people, how many did the test catch?”
        •	Out of 10 sick people, the test correctly identified 8.
        •	2 sick people slipped through (false negatives).

    Recall= TP/(TP + FN) = 8/(8+2) = 0.80

    ➡️ The test caught 80% of the sick people, but it missed 20%.
   

    🔹 Key Difference
        •	Precision cares about the quality of positive predictions.
            •	“When I say YES, how often am I right?”
        •	Recall cares about the coverage of actual positives.
            •	“Did I catch everyone who’s actually YES?”
    
    🔹 When do they differ?
        1.	Spam filter:
            •	Precision high → almost everything marked spam really is spam (few false alarms).
            •	Recall low → some spam still sneaks into inbox.
        2.	Medical test:
            •	Recall high → catches nearly all sick patients (few misses).
            •	Precision low → some healthy patients are incorrectly flagged.

    So they measure different trade-offs.
    
    ✅ Think of it this way:
        •	Precision = trust → Can I trust the “YES” predictions?
        •	Recall = coverage → Did I find all the actual “YES” cases?

'''