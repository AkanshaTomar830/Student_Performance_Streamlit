"""
MEMBER 2: Model Evaluation & Comparison Benchmarks
Evaluates the production Random Forest model and compares it with alternative algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Dict, Any


def evaluate_primary_model(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics for the production model on benchmark data.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    # cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y, y_pred) * 100
    prec = precision_score(y, y_pred) * 100
    rec = recall_score(y, y_pred) * 100
    f1 = f1_score(y, y_pred) * 100
    
    return {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1_score": round(f1, 2),
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": len(y)
    }


def benchmark_model_comparison(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> pd.DataFrame:
    """
    Trains and compares 3 standard classifiers using k-fold cross validation.
    Demonstrates scientifically why Random Forest is chosen as the champion model.
    """
    candidates = {
        "Random Forest (Champion)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
    }
    
    results = []
    for name, clf in candidates.items():
        scores_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy") * 100
        scores_f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1") * 100
        
        results.append({
            "Model": name,
            "CV Accuracy (%)": round(scores_acc.mean(), 2),
            "CV Std Dev (±)": round(scores_acc.std(), 2),
            "CV F1 Score (%)": round(scores_f1.mean(), 2)
        })
        
    return pd.DataFrame(results).sort_values(by="CV Accuracy (%)", ascending=False)
