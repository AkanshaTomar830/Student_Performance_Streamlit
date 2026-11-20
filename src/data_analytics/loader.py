"""
MEMBER 1: Dataset Analysis & Visualizations Module
Responsible for loading raw dataset, summary statistics, and data exploration.
"""

import pandas as pd
import os

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "student-mat.csv")


def load_student_dataset(filepath=None) -> pd.DataFrame:
    """
    Loads the student dataset from CSV.
    Default dataset is the UCI Mathematics course dataset.
    """
    if filepath is None:
        filepath = os.path.abspath(DEFAULT_DATA_PATH)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath, sep=";")
    # Ensure standard binary target column exists
    if "passed" not in df.columns and "G3" in df.columns:
        df["passed"] = (df["G3"] >= 10).astype(int)
        df["status"] = df["passed"].map({1: "Pass", 0: "Fail"})
    
    return df


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """
    Returns high-level statistics for dashboard metrics.
    """
    total_students = len(df)
    pass_count = (df["passed"] == 1).sum() if "passed" in df.columns else 0
    fail_count = total_students - pass_count
    pass_rate = (pass_count / total_students) * 100 if total_students > 0 else 0
    
    return {
        "total_students": total_students,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_rate, 1),
        "avg_g1": round(df["G1"].mean(), 2) if "G1" in df.columns else 0,
        "avg_g2": round(df["G2"].mean(), 2) if "G2" in df.columns else 0,
        "avg_g3": round(df["G3"].mean(), 2) if "G3" in df.columns else 0,
        "avg_absences": round(df["absences"].mean(), 1) if "absences" in df.columns else 0,
    }
