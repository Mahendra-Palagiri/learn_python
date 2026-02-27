''' Week 7 — Day 1: Bias–Variance Tradeoff (Symptoms + Diagnosis)

Topics
	•	Bias vs Variance (practical definitions)
	•	Underfitting vs Overfitting (train/validation symptoms)
	•	How model complexity shifts bias/variance:
	    •	linear vs polynomial degree
	    •	regularization strength (Ridge/Lasso)
	    •	tree depth / min samples
	•	Learning curves (train vs validation error vs training set size):
	    •	data scarcity vs model mismatch vs noise ceiling

🎯 Learning Goals
	•	Explain bias and variance in simple words
	•	Diagnose underfit vs overfit using train vs validation results
	•	Pick the correct “complexity knob” to turn (increase vs decrease complexity)
	•	Use learning-curve patterns to decide what’s limiting performance (data, model, or noise)
	
'''


# ==========================================================
# 1 — Bias vs Variance
# ==========================================================
'''
Bias
	•	Error from a model being too simple/too restricted to capture the real pattern
    •	How to recognize bias. --> "The model always missess the pattern in a similar way"

Variance
    •	Error from a model being too sensitive to the specific training data
    •	How to recognize Variance. --> "Learning noise and quirks not the real signal"

The variance of these factors
    •	High Bias --> Underfit
    •	High Variance --> Overfit
'''


# ==========================================================
# 2 — Symptoms (Train vs Validation)
# ==========================================================
'''
Train Performance       Validation Performance              What it means?
-----------------       ----------------------              ---------------
bad                     bad                                 Underfitting (high bias)
good                    bad                                 Overfitting (high variance)
good                    good (slightly worse than train)    Healthy/generalizing 


Important: 
    •	validation being a bit worse than train is normal. The problem is when the gap is large

Key instinct:
	•	Underfitting → model needs more capacity (or better features)
	•	Overfitting → model needs more constraint (regularization / simpler model) or more data
'''


# ==========================================================
# 3 — The "Complexity Knobs" (how we shift the tradeoff)
# ==========================================================
'''
A) Linear vs Polynomial degree
	•	Linear model (low complexity): tends to have higher bias, lower variance.
	•	Polynomial features (higher degree): increases complexity → can reduce bias but can increase variance (overfit risk).

    Rule:
    •	If underfitting → try adding polynomial degree (carefully)
    •	If overfitting → reduce degree or regularize

B) Regularization strength (Ridge/Lasso)

    Regularization = adding a penalty so the model doesn’t become too wild.
	•	Stronger regularization (higher alpha/lambda):
        •	reduces variance (less overfitting)
        •	may increase bias (can underfit if too strong)
	•	Weaker regularization:
        •	reduces bias (fits more)
        •	may increase variance (overfit risk)

    Ridge vs Lasso (one-liner):
	•	Ridge shrinks all coefficients
	•	Lasso can shrink some to zero (feature selection effect)

C) Tree depth / min samples
    Trees can become extremely flexible.
	•	Deeper tree (higher max_depth, smaller min_samples_leaf):
	    •	lower bias, higher variance (overfit risk)
	•	Shallower tree (lower max_depth, larger min_samples_leaf):
	    •	higher bias, lower variance (more stable)
'''

# ==========================================================
# 4 — Learning Curves. (Best diagnositc tool to understand these factors)
# ==========================================================
'''
Learning curves plot train error and validation error as training data increases.

We use them to answer: Is the limit coming from data, model simplicity, or unavoidable noise?

Pattern 1 — Model mismatch (high bias / underfitting)
	•	Train error: high
	•	Validation error: high
	•	Curves close together, both bad

> Meaning: model too simple / features not expressive 
> Typical fix: increase complexity (better features, polynomial/interactions, different model)

Pattern 2 — Overfitting (high variance)
	•	Train error: low
	•	Validation error: higher
	•	Noticeable gap

> Meaning: model too flexible for the data
> Typical fix: regularization, simpler model, constrain tree, or add more data

Pattern 3 — Data scarcity
	•	Validation improves as more data is added (gap may shrink)

> Meaning: more data likely helps

Pattern 4 — Noise ceiling
	•	Both curves flatten and stop improving past a point

> Meaning: data has noise/limits; improvements require better features, cleaner data, or reframing the problem

'''

#=========================================================
# Summarization in simple words
#=========================================================
'''
    Today we learned what bias and variance mean and how to recognize them from train vs validation behavior. 
    
    If both training and validation performance are bad, that indicates high bias (underfitting). 
    If training performance is good but validation performance is bad, that indicates high variance (overfitting). 
    
    We also learned the “knobs” to adjust complexity: 
        To reduce high bias we generally increase complexity gradually 
            (e.g., polynomial features, deeper trees, or weaker regularization). 
        To reduce high variance we generally reduce complexity 
            (e.g., simpler/linear models, shallower trees, or stronger regularization). 
    
    Finally, learning curves (train vs validation error as data increases) help decide whether the limitation is 
        data scarcity, 
        model mismatch/high bias, or 
        a noise ceiling (irreducible error).
'''