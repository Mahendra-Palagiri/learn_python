"""
Week 3 – Day 7: Iris Classifier Improvement Lab (Complete Solution)

Requirements:
    pip install scikit-learn numpy pandas matplotlib seaborn
Outputs:
    - week3_day7_results.csv
    - roc_curves.png
    - feature_importance_logreg.png
    - feature_importance_dtree.png
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ------------------------------
# Helpers
# ------------------------------

def micro_roc_auc_ovr(model, X_test, y_test) -> float:
    """Compute micro-average ROC-AUC (one-vs-rest) using predict_proba."""
    proba = model.predict_proba(X_test)
    # binarize labels into one-hot for OvR micro-average
    n_classes = proba.shape[1]
    y_onehot = np.eye(n_classes)[y_test]
    return roc_auc_score(y_onehot, proba, average="micro")


def plot_micro_roc_curves(models: Dict[str, Any], X_test, y_test, out_path="roc_curves.png"):
    """Plot micro-average ROC curves for multiple models on the same figure."""
    plt.figure(figsize=(8, 6))
    n_classes = len(np.unique(y_test))
    y_onehot = np.eye(n_classes)[y_test]

    for name, m in models.items():
        proba = m.predict_proba(X_test)
        # Micro-average: flatten true and scores
        fpr, tpr, _ = roc_curve(y_onehot.ravel(), proba.ravel())
        plt.plot(fpr, tpr, label=f"{name}")

    # Diagonal reference
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-Average ROC Curves (OvR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance_logreg(model: Pipeline, feature_names, out_path="feature_importance_logreg.png"):
    """Aggregate absolute logistic regression coefficients across classes and plot."""
    # model is a Pipeline([... ('model', LogisticRegression)])
    logreg: LogisticRegression = model.named_steps["model"]
    coefs = np.abs(logreg.coef_)            # shape: (n_classes, n_features)
    agg = coefs.sum(axis=0)                 # aggregate importance per feature
    order = np.argsort(agg)[::-1]
    plt.figure(figsize=(7, 4))
    plt.bar([feature_names[i] for i in order], agg[order])
    plt.title("Logistic Regression: Aggregated |coef_| per feature")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance_tree(model: Pipeline, feature_names, out_path="feature_importance_dtree.png"):
    """Plot Decision Tree feature_importances_."""
    tree: DecisionTreeClassifier = model.named_steps["model"]
    imp = tree.feature_importances_
    order = np.argsort(imp)[::-1]
    plt.figure(figsize=(7, 4))
    plt.bar([feature_names[i] for i in order], imp[order])
    plt.title("Decision Tree: feature_importances_")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def summarize_model(name: str, model, X_test, y_test) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    roc = micro_roc_auc_ovr(model, X_test, y_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    return {
        "Model": name,
        "Accuracy": acc,
        "F1_macro": f1,
        "ROC_AUC_micro_OvR": roc,
        "Report": report
    }


# ------------------------------
# Main
# ------------------------------

def main(random_state: int = 42):
    # 1) Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    # 3) Pipelines
    # KNN & Logistic Regression benefit from scaling; DecisionTree does not need it.
    pipe_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ])

    pipe_logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, multi_class="auto"))
    ])

    pipe_tree = Pipeline([
        ("model", DecisionTreeClassifier(random_state=random_state))
    ])

    # 4) Hyperparameter search spaces
    # GridSearch for KNN
    param_grid_knn = {
        "model__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2]  # Manhattan vs Euclidean
    }
    grid_knn = GridSearchCV(
        pipe_knn, param_grid_knn, cv=5, n_jobs=-1, scoring="accuracy"
    )

    # GridSearch for Logistic Regression
    param_grid_logreg = {
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"]  # good default for multi_class
    }
    grid_logreg = GridSearchCV(
        pipe_logreg, param_grid_logreg, cv=5, n_jobs=-1, scoring="accuracy"
    )

    # RandomizedSearch for Decision Tree (example of randomized tuning)
    param_dist_tree = {
        "model__max_depth": [None] + list(range(2, 11)),
        "model__min_samples_split": [2, 5, 10, 15],
        "model__min_samples_leaf": [1, 2, 4, 6, 8],
        "model__criterion": ["gini", "entropy", "log_loss"]
    }
    rnd_tree = RandomizedSearchCV(
        pipe_tree, param_distributions=param_dist_tree,
        n_iter=20, cv=5, n_jobs=-1, random_state=random_state, scoring="accuracy"
    )

    # 5) Fit searches
    grid_knn.fit(X_train, y_train)
    grid_logreg.fit(X_train, y_train)
    rnd_tree.fit(X_train, y_train)

    best_knn = grid_knn.best_estimator_
    best_logreg = grid_logreg.best_estimator_
    best_tree = rnd_tree.best_estimator_

    # 6) Evaluate and print reports
    results = []
    for name, model, search in [
        ("KNN", best_knn, grid_knn),
        ("LogisticRegression", best_logreg, grid_logreg),
        ("DecisionTree", best_tree, rnd_tree),
    ]:
        summary = summarize_model(name, model, X_test, y_test)
        summary["BestParams"] = search.best_params_
        results.append(summary)

        print("=" * 60)
        print(f"{name} | Best Params: {search.best_params_}")
        print(f"Accuracy: {summary['Accuracy']:.3f} | "
              f"F1_macro: {summary['F1_macro']:.3f} | "
              f"ROC_AUC_micro_OvR: {summary['ROC_AUC_micro_OvR']:.3f}")
        print("Classification Report:")
        print(summary["Report"])

    # 7) Results table -> CSV
    df_rows = []
    for r in results:
        df_rows.append({
            "Model": r["Model"],
            "BestParams": r["BestParams"],
            "Accuracy": r["Accuracy"],
            "F1_macro": r["F1_macro"],
            "ROC_AUC_micro_OvR": r["ROC_AUC_micro_OvR"]
        })
    df = pd.DataFrame(df_rows).sort_values(by="Accuracy", ascending=False)
    df.to_csv("week3_day7_results.csv", index=False)
    print("\nSaved summary to week3_day7_results.csv")
    print(df)

    # 8) Plot ROC curves (micro-average per model)
    plot_micro_roc_curves(
        {
            "KNN": best_knn,
            "LogReg": best_logreg,
            "DecisionTree": best_tree
        },
        X_test, y_test, out_path="roc_curves.png"
    )
    print("Saved micro-average ROC curves to roc_curves.png")

    # 9) Feature importance plots for LogReg and Tree
    plot_feature_importance_logreg(best_logreg, feature_names, out_path="feature_importance_logreg.png")
    print("Saved logistic-regression feature importances to feature_importance_logreg.png")

    plot_feature_importance_tree(best_tree, feature_names, out_path="feature_importance_dtree.png")
    print("Saved decision-tree feature importances to feature_importance_dtree.png")

    # 10) Final winner
    winner_row = df.iloc[0]
    print("\n🏆 Best overall (by Accuracy):")
    print(winner_row.to_string(index=False))


if __name__ == "__main__":
    main()