''' *** Week 5 · Day 5 — Regularization (L1 & L2) in Logistic Regression

🎯 Learning Goal

By the end of today, we will clearly understand:
	•	Why unregularized logistic regression is dangerous
	•	What overfitting really means in coefficient space
	•	How L2 (Ridge) and L1 (Lasso) behave differently
	•	What the parameter C actually controls
	•	Why regularization is not optional in real systems

'''

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

titdf = pd.read_csv('./data/week5/titanic_synthetic.csv')

num_cols = ['Age','Fare']
cat_cols = ['Embarked','Sex']
features = num_cols + cat_cols
target  = 'Survived'

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scaler',RobustScaler())
])

cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(handle_unknown='ignore'))
])

pre_prcsr = ColumnTransformer([
    ('nums',num_pipe,num_cols),
    ('catg',cat_pipe,cat_cols)
])


X = titdf[features]
Y = titdf[target]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#L2 Regularization
model_l2 = LogisticRegression(
    penalty='l2',
    C=1,
    max_iter=1000,
    solver='lbfgs'
)

l2_pipe = Pipeline([
    ('preprcsr',pre_prcsr),
    ('model',model_l2)
])

l2_pipe.fit(X_train,Y_train)

l2_feature_names = l2_pipe.named_steps['preprcsr'].get_feature_names_out()
l2_coef= l2_pipe.named_steps['model'].coef_[0]

l2_coef_df  = (
    pd.DataFrame({'features':l2_feature_names, 'coef':l2_coef}).sort_values(by='coef', key=abs, ascending=False)
)
# print(l2_coef_df)


#L1 Regularization
model_l1 = LogisticRegression(
    penalty='l1',
    C=1,
    max_iter=1000,
    solver='liblinear'
)

l1_pipe = Pipeline([
    ('preprcsr',pre_prcsr),
    ('model',model_l1)
])

l1_pipe.fit(X_train,Y_train)

l1_feature_names = l1_pipe.named_steps['preprcsr'].get_feature_names_out()
l1_coef= l1_pipe.named_steps['model'].coef_[0]

l1_coef_df  = (
    pd.DataFrame({'features':l1_feature_names, 'coef':l1_coef}).sort_values(by='coef', key=abs, ascending=False)
)
# print('\n\n\n\n',l1_coef_df)


#Varying C
for c in [0.01,0.1,1,30,100]:
    model = LogisticRegression(
        penalty='l1',
        C=c,
        max_iter=1000,
        solver='liblinear'
    )

    c_pipe = Pipeline([
        ('preprcsr',pre_prcsr),
        ('model',model)
    ])

    c_pipe.fit(X_train,Y_train)
    cYpred = c_pipe.predict(X_test)
    print(f'\n C={c} → non-zero coefficients: '
          ,(c_pipe.named_steps['model'].coef_[0])
          ,'\n Feature Names: ',l1_feature_names
          , '\n Accuracy Score --> ', accuracy_score(Y_test,cYpred) 
          ,'\n Classification Report --> \n',classification_report(Y_test,cYpred)
          ,'\n\n\n'
          )


'''  ** Theory
1️⃣ Why Regularization Exists (the real reason)

Recall the logistic regression objective:

Loss = Log Loss

Unregularized logistic regression tries to:

Push coefficients as far as needed to reduce classification error.

Problem:

If features are:
	•	correlated
	•	noisy
	•	weakly predictive

the model can:
	•	inflate coefficients
	•	become unstable
	•	overfit noise
	•	behave badly on new data

This is not theoretical — it happens all the time.

⸻

2️⃣ Regularization = Constraint on Coefficients

Regularization adds a penalty term:

** L2 (Ridge)
-------------
ƛ --> lambda
∑ --> sum

Loss = Log Loss + ƛ ∑ w^2

** L1 (Lasso)
-------------
|w| --> Absolute value of weight

Loss = Log Loss + ƛ ∑ |w|


Interpretation (important):

Regularization tells the model:
“Fit the data — but don’t be overly confident unless you really must.”

⸻

3️⃣ What C Actually Means (this is critical)

In scikit-learn:
LogisticRegression(C=1.0)

C is inverse regularization strength:

C value             Meaning
--------            --------
Small C (0.01)      Strong regularization
Medium C (1.0)      Balanced
Large C (100)       Almost no regularization

So:
	•	Lower C → simpler model
	•	Higher C → more flexible model

This is the opposite of how λ is written in math books — keep that straight.

⸻

4️⃣ L2 Regularization (Ridge): “Shrink, don’t kill”

Behavior:
	•	Shrinks all coefficients
	•	Keeps all features
	•	Rarely sets anything to exactly zero

Intuition:

“Every feature can contribute — but none too much.”

When L2 is preferred:
	•	Correlated features
	•	When you believe most features matter
	•	When stability matters more than sparsity

⸻

5️⃣ L1 Regularization (Lasso): “Select, then fit”

Behavior:
	•	Drives some coefficients to exactly zero
	•	Performs implicit feature selection

Intuition:

“Only a few features deserve to exist.”

When L1 is preferred:
	•	High-dimensional data
	•	Many irrelevant features
	•	You want interpretability
	•	You want sparsity

6️⃣ Key Conceptual Takeaways
	•	Regularization controls model confidence
	•	L2 = smooth, stable, conservative
	•	L1 = aggressive, selective, sparse
	•	C controls how much the model is allowed to “believe itself”
	•	Regularization changes interpretation of coefficients, not just accuracy

        Q)  we didnt see any change in behvaior espeically accuracy score and classification report (precision, recall, f1 score) with varying C and l1 penalty for the model) why is that
        egularization strongly affects coefficients, but often has little or no effect on accuracy / precision / recall — especially in simple, well-specified problems.

        Let’s unpack this carefully.

        ⸻

        1️⃣ Key principle (anchor this)

        Metrics measure predictions.
        Regularization mainly reshapes coefficients.

        If reshaping coefficients does not change the predicted probabilities enough to cross decision thresholds, then:
            •	accuracy
            •	precision
            •	recall
            •	F1

        will remain almost identical.

        That’s exactly what you saw.

        ⸻

        2️⃣ Why this happened in your case (specific, not generic)

        Your setup has these properties:

        ✔ Few features
            •	Age
            •	Fare
            •	Sex
            •	Embarked

        This is a low-dimensional problem.

        ⸻

        ✔ Strong, clean signal
            •	Sex is dominant
            •	Embarked_C is clear
            •	Age & Fare are consistent

        There is no feature explosion, no noise swamp.

        ⸻

        ✔ Balanced dataset
            •	~50/50 survival
            •	No extreme class imbalance
            •	Default threshold works well

        ⸻

        ✔ Logistic regression already near optimal

        Even unregularized, the model:
            •	is not overfitting
            •	is not unstable
            •	is not memorizing noise

        So when you add L1/L2:
            •	coefficients shrink
            •	some drop out
            •	but decision boundaries barely move

        ⸻

        3️⃣ Why predictions didn’t change even when coefficients did

        This is subtle but crucial.

        Logistic regression predicts:

        \sigma(w_0 + w_1 x_1 + w_2 x_2 + \dots)

        If:
            •	coefficients shrink proportionally
            •	dominant features remain dominant
            •	relative ordering of samples stays the same

        Then:

        Probabilities shift slightly, but rankings stay the same.

        And if rankings stay the same:
            •	ROC stays the same
            •	Threshold-based predictions stay the same
            •	Metrics stay the same

        This is why:
            •	AUC barely moved
            •	Accuracy barely moved
            •	Precision/Recall barely moved

        ⸻

        4️⃣ Why this is actually a good sign

        Many learners expect:

        “Regularization should improve accuracy”

        That is not the main purpose of regularization.

        Regularization’s real goals:
            •	improve generalization
            •	improve stability
            •	prevent coefficient explosion
            •	handle future noise

        Not necessarily:
            •	improve training or test accuracy

        In clean problems, regularization is preventive, not corrective.

        ⸻

        5️⃣ When you would see metric changes

        You would see noticeable metric changes if any of these were true:

        🔥 High dimensionality
            •	100s / 1000s of features

        🔥 Strong multicollinearity
            •	Many correlated numeric variables

        🔥 Noisy or weak signals
            •	True signal barely above noise

        🔥 Severe class imbalance
            •	Threshold sensitivity amplified

        🔥 Very small dataset
            •	Overfitting becomes visible

        Your current dataset has none of these problems.

        So the lack of metric change is expected and correct.

        ⸻

        6️⃣ L1 vs L2: why metrics stayed similar even when features dropped

        You saw:
            •	L1 dropped features
            •	L2 kept all features
            •	Metrics stayed similar

        Why?

        Because:
            •	Dropped features were redundant
            •	Surviving features captured most of the signal
            •	Decision surface didn’t change meaningfully

        This is a classic case of:

        Multiple models with different explanations but similar predictive power

        That’s very common in ML.

        ⸻

        7️⃣ This is an important real-world lesson

        Do not judge regularization by accuracy alone.

        Regularization is about:
            •	robustness
            •	interpretability
            •	stability under data shift
            •	confidence control

        A model that:
            •	performs the same today
            •	but is more stable tomorrow

        is a better model, even if metrics are identical.

        ⸻

        8️⃣ One-line summary (bookmark this)

        Regularization changes how the model reasons, not necessarily what it predicts — especially when the problem is simple and well-specified.

'''