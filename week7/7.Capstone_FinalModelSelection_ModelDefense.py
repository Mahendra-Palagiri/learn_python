'''Week 7 — Day 7: Capstone — Final Model Selection + Model Defense

📚 Topics
	•	Final selection workflow (selection vs assessment discipline)
	•	Evidence-based comparison (CV mean + std + simplicity)
	•	“Model Defense” write-up (why this model, what risks remain)
	•	Final checklist: leakage, metric choice, stability, next improvements

🎯 Learning Goals
	•	Choose a final model using evidence (not vibes)
	•	Identify next steps to improve performance (features/model family/thresholding)
	•	Lock a repeatable template for future projects

'''


#=========================================================
# Week 7. --> RetroSpection
#=========================================================
'''
1) Decision summary (based on Day 4–6 evidence)

Candidate models tested (5-fold Stratified CV, F1):
	•	LogReg: mean 0.6495, std 0.0132
	•	DecisionTree: mean 0.5536, std 0.0069
	•	RandomForest: mean 0.5913, std 0.0070

Final choice: LogisticRegression Pipeline
Reason: Highest mean F1 by a meaningful margin. Stability is acceptable. Also simplest to deploy and explain.

⸻

2) Final model configuration (best from Day 5)

Best params found:
	•	C = 0.01, penalty = l1, class_weight = None, scaler = StandardScaler

So the final pipeline is:
	•	Numeric: median impute → StandardScaler
	•	Categorical: most_frequent impute → OneHotEncoder
	•	Model: LogisticRegression (liblinear, L1, C=0.01)

⸻

3) Model Defense 
We selected a leakage-safe Pipeline (imputation + scaling/encoding + classifier) 
and used 5-fold Stratified cross-validation with F1 as the metric. 

Logistic Regression achieved the best average F1 (~0.65) 
compared to Decision Tree (~0.55) and Random Forest (~0.59). 

The CV standard deviation (~0.013) indicates reasonably stable performance across folds. 
Hyperparameter tuning (C, penalty, class_weight, scaler choice) did not materially improve results, 
suggesting the current performance is limited more by feature signal/noise than by tuning. 

The final model is simple, interpretable, and consistent, making it the best choice for this dataset 
under the current feature set.

⸻

4) Risks and what to monitor
	•	Noise ceiling / limited signal: tuning didn’t help much → likely feature-limited
	•	Synthetic dataset limitation: CV performance might not transfer to real Titanic distribution
	•	Threshold sensitivity: F1 depends on decision threshold (0.5 may not be optimal)
	•	Data drift: if class proportions change, F1 behavior changes

⸻

5) Next improvements (practical)
	•	Add stronger predictive features (if available): family size, title extraction, ticket group features (for real Titanic)
	•	Try threshold tuning for F1 (choose threshold that maximizes CV F1)
	•	Try gradient boosting models (XGBoost/LightGBM-style equivalents in sklearn: HistGradientBoosting) using same pipeline discipline
	•	Error slicing: evaluate F1 by subgroup (Sex, Cabin missing vs present, Embarked)
'''