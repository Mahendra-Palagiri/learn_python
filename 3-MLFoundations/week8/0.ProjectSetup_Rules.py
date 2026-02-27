'''Week 8 Kickoff — Phase 0: Project Setup & Rules

📚 Topics
	•	Capstone scope selection (task + dataset)
	•	Evaluation rules (leakage prevention)
	•	Project skeleton + repo hygiene
	•	Metric commitment (primary + secondary)

🎯 Learning Goals
	•	We lock the scope and the rules before writing model code.
	•	We create the project folder structure and the minimal files we will use all week.
	•	We commit to a primary metric early and document it.

'''

#============================================================================
# Create Folder structure and basic files we are going to use for the project
#===========================================================================
# mkdir -p week8/{data,notebooks,src,reports}
# touch week8/src/{config.py,data_load.py,pipeline.py,train.py,evaluate.py}
# touch week8/reports/{model_comparison.md,model_defense.md}


#============================================================================
# Capstone Track
#===========================================================================
'''
	•	Task: Classification
	•	Baseline Model: LogisticRegression
	•	Candidate Models: RandomForestClassifier + (optional) GradientBoostingClassifier
	•	Primary Metric: ROC-AUC (works well when we want threshold-independent ranking)
	•	Secondary Metrics: Accuracy, Precision, Recall, F1

    If the dataset is imbalanced later, 
    we may switch the primary metric to PR-AUC, 
    but we do not change metrics mid-stream without documenting why.
'''