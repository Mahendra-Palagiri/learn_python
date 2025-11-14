''' Week 5 — Day 1: Logistic Regression Fundamentals

🎯 Learning Goal

Build a deep mathematical and intuitive understanding of logistic regression 
    - what it is 
    - Why it’s used for classification, and 
    - How the sigmoid (logistic) function transforms linear outputs into probabilities.

    
🔍 1. What is Classification?

    Before we dive into math, frame the problem.
    Regression                          Classification
    ---------------                     -----------------
    Predicts continuous values          Predicts categories (0/1, yes/no, survive/die)
    Uses linear regression              Uses logistic regression
    Output is unbounded real numbers    Output is probability between 0 and 1


🧠 2. The Core Idea Behind Logistic Regression
    - Logistic regression does not predict classes directly.
    - It predicts the probability of belonging to class 1.

    - It does that in 3 Stages
        Features → Linear combination (z) → Sigmoid → Probability → Prediction

    P(Y=1|X) = sigmoid(wX + b)

    The linear expression:
    z = wX + b
    z = w_0 + w_1 . Age + w_2 . Fare + w_3 . Sex_female + w_4 . Embarked_C + …. 
        - All the features are being consdiered to arrive at singular number
        - The weights w_2, w_3 ... are the coeffients of the features after applying logisticregression model (which we have explored in previous weeks)
        - The wight w_0 aka as 'the intercept' is an output of logisticregression model

            model = LogisticRegression()
            model.fit(X, y)

            model.coef_      # → weights (one per feature)
            model.intercept_ # → bias term (w0)

    can be anywhere: large, small, positive, negative.

    But probabilities must be:
    0 <= 'p' <= 1

    So we apply a squashing function.

⸻

🌀 3. The Sigmoid (Logistic) Function

    P= 𝛔(z) = 1/(1 + e^-z)

    Properties:
        •	If z → +∞ → probability → 1
        •	If z → –∞ → probability → 0
        •	At z = 0 → probability = 0.5

    ```python
    import numpy as np

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    print(sigmoid(np.array([-10, -1, 0, 1, 10])))
    ```

📈 4. Odds and Log-Odds (The Heart of Logistic Regression)

    Odds:
        Odds = P/1-P

    Log-Odds (logit)
        logit(P) = ln(P/1-P)

    Logistic regression assumes:
        logit(P) = wX + b

    Meaning:
        •	The relationship between X and log-odds is linear
        •	The relationship between X and probability is nonlinear

    This is the most important conceptual insight of Day 1.

📗 5. Why Logistic Regression Is Linear (But Probabilistic)

    Step    Explanation
    ----    -----------
    1       Model forms a linear boundary in log-odds space
    2       Sigmoid squeezes values into probabilities
    3       Threshold (default = 0.5) converts probability → class

    Thus, logistic regression is:
    Linear in parameters → Nonlinear in prediction space

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


titdf = pd.read_csv('./data/week5/titanic_synthetic.csv')
# print("\n\n --> \n",titdf.info())
# print("\n\n --> \n",titdf.describe(include='all'))


def sigmoid(z):
        return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
p = sigmoid(z)

plt.plot(z, p)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Probability")
plt.grid(True)
plt.show()


