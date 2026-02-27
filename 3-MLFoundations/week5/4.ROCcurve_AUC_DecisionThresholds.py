''' *** Week 5 · Day 4 — ROC Curve, AUC & Decision Thresholds
🎯 Learning Goal

Today we will understand:
	•	Why classification ≠ fixed threshold. (Predict vs Predict_Proba)
	•	How to evaluate a classifier independent of thresholds
	•	What ROC and AUC really measure
	•	How changing thresholds reshapes precision & recall
	•	Why Day 3 metrics are just one operating point

    This is the conceptual bridge from “metrics” → “model behavior”.
'''

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,classification_report, roc_auc_score
import matplotlib.pyplot as plt

titdf = pd.read_csv('./data/week5/titanic_synthetic.csv')
# print(titdf.info())
# print(titdf.describe(include='all'))

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

preprocess = ColumnTransformer([
    ('nums',num_pipe,num_cols),
    ('catg',cat_pipe,cat_cols)
])

final_pipe = Pipeline([
    ('preprcsr',preprocess),
    ('model',LogisticRegression(max_iter=1000))
])

X = titdf[features]
Y = titdf[target]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

final_pipe.fit(X_train,Y_train)
y_proba = final_pipe.predict_proba(X_test)[:,1]
# print('\n\nY_PROBA --> ',y_proba.min(), y_proba.max(),'\n\n')

finalYpred = final_pipe.predict(X_test) #Under the hood Predit uses (y_proba >= 0.5).astype(int) i.e. its considering only one threshold of 0.5 
acc = accuracy_score(Y_test,finalYpred)
cm = confusion_matrix(Y_test,finalYpred)
# print(cm)

clrpt = classification_report(Y_test,finalYpred)
# print(clrpt)



''' ROC Curve — What It Actually Represents
Definitions
	•	TPR (Recall)
        TPR = TP/(TP + FN)
	•	FPR
        FPR = FP/(FP + TN)

    Each threshold gives:
        •	one TPR
        •	one FPR
        •	one point on the ROC curve
'''
fpr, tpr, thresholds = roc_curve(Y_test,y_proba)
# print(thresholds[0], '\n\n Last threshold. --> ', thresholds[-1]) #First and last element values

plt.plot(fpr,tpr, label='Logistic Regression')
plt.plot([0,1],[0,1],'k--',label='Random')
plt.xlabel('False Postivie Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
# plt.show()

rocauc = roc_auc_score(Y_test,y_proba)
print(rocauc) # Output value 0.661006906808626

for t in [0.7,0.6,0.66,0.5,0.4,0.3]:
    y_pred_t = (y_proba >= t).astype(int)
    print(f"\nThreshold = {t}")
    print(classification_report(Y_test, y_pred_t))

#Observation
	# •	Lower threshold → higher recall for class 1
	# •	Lower threshold → lower precision for class 1
	# •	F1 peaks somewhere in the middle


''' --- RETROPSECTION ---
***** 
is roc_auc_score the min or max threshold that can be used i.e. in our example our roc_auc_score is 0.66 is that why when we were generating classification reports with diff thresholds we started with 0.6?
This is a very important clarification, because confusing AUC with thresholds is a common but serious misunderstanding.

Short answer first, then the deep explanation:

NO — roc_auc_score has nothing to do with thresholds.
0.66 is NOT a threshold.
It does NOT constrain which thresholds you can use.

Now let’s unpack this properly.

⸻

1️⃣ What AUC actually is

roc_auc_score returns Area Under the ROC Curve.

That number answers one question only:

If I randomly pick one positive and one negative example,
what is the probability that the model ranks the positive higher than the negative?

So:
	•	AUC = 0.66 means
→ In 66% of random (positive, negative) pairs,
→ the model assigns a higher probability to the positive.

That’s it.

It is:
	•	❌ NOT a probability threshold
	•	❌ NOT a cutoff
	•	❌ NOT a min or max allowed value


*****
explain with a detailed example when precision matters , when recall matters if F1 is a mean average why does it matter at all. 
(with diff thresholds the binary classes of survived or not survied is changing on these metrics)

The Core Problem (before metrics)

You have a model that outputs:

Probability that a passenger survived

But probabilities are not decisions.

To make a decision, you must choose a threshold.

Every threshold creates a different world:
	•	who is predicted “survived”
	•	who is predicted “not survived”
	•	how many mistakes of each type you make

Metrics exist to describe which mistakes you care about.

⸻

<> Step 1: The Two Types of Mistakes (this is everything)

In binary classification there are only two possible errors:
Error                   Meaning
False Positive (FP)     You predicted Survived, but the passenger did not survive
False Negative (FN)     You predicted Did not survive, but the passenger did survive

Every metric is just a different way of saying:

Which mistake hurts more?

<> Step 2: Precision — “Don’t Cry Wolf” Metric

Definition (in plain English)

Precision answers:
“When the model says Survived, how often is it correct?”

Formula

Precision = TP/(TP + FP)

⸻

When Precision Matters (detailed examples)

🚨 Example 1: Medical Alert System (False Positives are expensive)

Imagine:
	•	Model flags patients as having cancer
	•	Each positive prediction triggers:
	•	biopsies
	•	stress
	•	cost
	•	potential harm

If precision is low:
	•	Many healthy patients are told they might have cancer
	•	Massive unnecessary harm

➡️ You want:
	•	High precision
	•	You are OK missing some real cases (FN)
	•	You want to be very sure when you raise an alarm

How threshold behaves here
	•	Use HIGH threshold (e.g., 0.8)
	•	Only very confident cases are predicted positive
	•	Fewer FP → higher precision
	•	More FN → lower recall

⸻

<> Step 3: Recall — “Don’t Miss Anyone” Metric

Definition (in plain English)

Recall answers:
“Of all actual survivors, how many did the model catch?”

Formula

Recall = TP/(TP + FN)

⸻

When Recall Matters (detailed examples)

🧯 Example 2: Fire Detection System (False Negatives are catastrophic)

Imagine:
	•	Model detects fire in a building
	•	Missing a fire = people die

If recall is low:
	•	Real fires go undetected
	•	Catastrophic outcome

➡️ You want:
	•	High recall
	•	You accept false alarms (FP)
	•	Better to evacuate unnecessarily than miss a fire

How threshold behaves here
	•	Use LOW threshold (e.g., 0.2)
	•	Many positives detected
	•	FN ↓ → recall ↑
	•	FP ↑ → precision ↓

⸻

<> Step 4: Why Precision and Recall Fight Each Other

This is not a bug — it is fundamental.
Lower threshold     Higher threshold
--------------      ----------------
More positives      Fewer positives
FP ↑                FP ↓
FN ↓                FN ↑
Recall ↑            Precision ↑

You cannot maximize both at the same time unless the problem is perfectly separable (almost never happens).
⸻

<> Step 5: Where F1 Comes In (and why it matters)

What F1 actually is

F1 = 2 x (Precision x Recall)/(Precision + Recall)


This is not a simple average.

It is a harmonic mean, which:
	•	punishes extreme imbalance
	•	drops sharply if either precision or recall is low

⸻

Why not just accuracy?

Because accuracy hides imbalance.

Example:
	•	95% of passengers died
	•	Model predicts “died” for everyone
	•	Accuracy = 95%
	•	Precision/Recall for survivors = 0

Accuracy lies.

⸻

Why F1 matters (deep reason)

F1 answers:

“Can this model make useful positive predictions?”

	•	If precision = 1.0 and recall = 0.0 → F1 = 0
	•	If recall = 1.0 and precision = 0.0 → F1 = 0
	•	Only when both are reasonably good does F1 rise

So F1 filters out:
	•	“I never predict positives”
	•	“I predict everything positive”

Both are useless.

⸻

Step 6: Bring it back to Titanic example

Our model outputs probabilities between:

0.28 → 0.78

Threshold = 0.6
	•	Few predicted survivors
	•	Precision ↑
	•	Recall ↓

Threshold = 0.3
	•	Many predicted survivors
	•	Recall ↑
	•	Precision ↓

Threshold ≈ 0.45
	•	Balanced behavior
	•	Precision ≈ Recall
	•	F1 peaks

That’s why threshold tuning directly changes these metrics.

⸻

Step 7: Which metric should YOU care about?

It depends entirely on problem cost:
Problem                         Metric Priority
---------                       ----------------
Medical diagnosis               Recall (don’t miss cases)
Fraud detection                 Precision (don’t block legit users)
Spam filtering                  Precision
Safety alarms                   Recall
Balanced business decision      F1
Ranking users                   AUC

There is no universally “best” metric.

⸻

Final Mental Model
Precision = “When I speak, am I right?”
Recall = “Did I miss anyone?”
F1 = “Am I useful at all?”
Threshold = “How bold do I want to be?”

Metrics don’t judge the model.
They describe the consequences of your threshold choice.

'''