"""
MEMBER 1: Preprocessing & Data Cleaning Module
Prepares feature matrices, handles encodings, and computes correlation matrices.
"""

import pandas as pd
from typing import Tuple, List

CORE_FEATURES = ["studytime", "failures", "absences", "G1", "G2"]


def prepare_features_and_target(df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts the feature matrix X and binary target vector y (1 = Pass, 0 = Fail).
    """
    if feature_cols is None:
        feature_cols = CORE_FEATURES
    
    # Verify columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")
    
    X = df[feature_cols].copy()
    
    if "passed" in df.columns:
        y = df["passed"].copy()
    elif "G3" in df.columns:
        y = (df["G3"] >= 10).astype(int)
    else:
        raise ValueError("Dataset does not contain 'passed' or 'G3' target column.")
    
    return X, y


def calculate_correlations(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Calculates the correlation matrix for key numeric performance variables.
    """
    if columns is None:
        columns = ["G1", "G2", "G3", "studytime", "failures", "absences", "age"]
    
    available_cols = [c for c in columns if c in df.columns]
    corr_matrix = df[available_cols].corr().round(2)
    return corr_matrix
