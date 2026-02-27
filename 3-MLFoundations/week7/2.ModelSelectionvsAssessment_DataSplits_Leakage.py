''' Week 7 — Day 2: Model Selection vs Model Assessment + Data Splits & Leakage

📚 Topics
	•	Model selection vs model assessment (tune vs report)
	•	Correct splitting patterns:
        •	Train/Validation/Test
        •	Cross-validation for tuning + final holdout test
	•	Data leakage (what it is, why it breaks truth)
	•	Common leakage patterns (the usual “gotchas”)
	•	The “golden rule”: preprocessing/feature decisions must happen inside the training-only process

🎯 Learning Goals
	•	Clearly state the difference between selection and assessment
	•	Explain why tuning on the test set is invalid (even “just once”)
	•	Identify at least 4 common leakage sources
	•	Describe a clean evaluation workflow we’ll use going forward

'''

# ==========================================================
# 1 — Model selection vs model assessment (the big separation)
# ==========================================================
'''
	•	Model selection = deciding which model + which hyperparameters to use.
            Output: “This is the chosen pipeline/config.”
	•	Model assessment = estimating how well the final chosen approach performs on unseen data.
            Output: “This is the performance we can honestly report.”

    Why this matters: if we use the same data to choose and judge, 
    we accidentally reward luck and overfitting to the evaluation itself.

'''

# ==========================================================
# 2 — Splitting patterns we’ll follow
# ==========================================================

'''
    Pattern A — Classic (simple projects)
        1.	Split once into train and test
        2.	Inside train: split again into train and validation (or use CV)
        3.	Use validation/CV for selection
        4.	Touch test only once at the end for assessment

    Pattern B — Best practice (most ML workflows)
        1.	Split into train and test
        2.	Use cross-validation on train for selection + tuning
        3.	Refit best model on full train
        4.	Evaluate once on test

    The key idea: test set is the “final exam,” not the practice quiz.
'''


# ==========================================================
# 3 — Data leakage (the silent performance inflator)
# ==========================================================
'''
Leakage =   using information during training (directly or indirectly) 
            that would not be available at prediction time, 
            or letting validation/test influence training decisions.

Result: performance looks amazing… then collapses in real life
'''

# ==========================================================
# 4 — Common leakage examples (must recognize)
# ==========================================================
'''
	•	Scaling/normalizing before splitting (validation stats leak into training)
	•	Encoding categories before splitting (validation categories/statistics leak)
	•	Imputation using whole dataset (mean/median computed using validation/test)
	•	Feature selection using full data (correlation, mutual info, p-values done on all data)
	•	Target leakage features:
	•	“future” information (post-outcome fields)
	•	aggregates that include the target period
	•	Time series shuffle leakage (training sees future patterns)

    Core fix (simple rule):

    Anything that learns from data must be fit using training-only, ideally via a Pipeline.
'''

# ==========================================================
# 5 — The clean workflow (what we’ll use going forward)
# ==========================================================
'''
	1.	Decide metric + baseline
	2.	Split train/test
	3.	Build Pipeline (prep + model)
	4.	Tune using CV on train
	5.	Pick best based on CV mean + stability
	6.	Final one-time score on test

'''


#=========================================================
# Summarization in simple words
#=========================================================
'''
    Today we understood model selection vs model assessment. 
    
    Model selection is choosing the model and hyperparameters using validation data or cross-validation on the training set. 
    Model assessment is evaluating the final chosen model on the held-out test set, ideally only once. 
    
    This separation prevents data leakage / overfitting to the test set and gives a more honest estimate of real-world performance

    We also learned common causes of data leakage: 
        doing encoding, imputation, or scaling before splitting, and performing feature selection using the full dataset. 
    
    If test data influences any preprocessing, feature decisions, or tuning, 
    the evaluation becomes contaminated—performance looks better than it truly is, 
    and the model can overfit to the evaluation, 
    which often leads to weaker performance in real-world unseen data
'''