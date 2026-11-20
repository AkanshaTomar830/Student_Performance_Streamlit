"""
MEMBER 2: Model Inference & Prediction Engine
Encapsulates prediction, confidence calculation, risk tiering, and scenario simulation.
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Models", "student_prediction_app_model.pkl")
DEFAULT_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Models", "app_features.pkl")


class StudentPredictor:
    """
    Production predictor wrapper around the trained Random Forest Classifier.
    """
    def __init__(self, model_path=None, features_path=None):
        self.model_path = os.path.abspath(model_path or DEFAULT_MODEL_PATH)
        self.features_path = os.path.abspath(features_path or DEFAULT_FEATURES_PATH)
        self.model = joblib.load(self.model_path)
        self.features = joblib.load(self.features_path)
        
        self.pass_index = list(self.model.classes_).index(1)
        self.fail_index = list(self.model.classes_).index(0)

    def predict_student(self, student_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference for an individual student dictionary.
        Returns outcome, probabilities, risk classification, and trajectory.
        """
        df = pd.DataFrame([student_dict])[self.features]
        prediction = int(self.model.predict(df)[0])
        probabilities = self.model.predict_proba(df)[0]
        
        pass_prob = round(float(probabilities[self.pass_index]) * 100, 1)
        fail_prob = round(float(probabilities[self.fail_index]) * 100, 1)
        
        # Risk tiering
        if pass_prob >= 75:
            risk_level = "Low Risk (Safe)"
            risk_badge = "Safe"
        elif pass_prob >= 50:
            risk_level = "Medium Risk (Borderline)"
            risk_badge = "Watchlist"
        else:
            risk_level = "High Risk (Critical)"
            risk_badge = "Critical"
        
        # Grade momentum (Exam 2 vs Exam 1)
        g1 = student_dict.get("G1", 10)
        g2 = student_dict.get("G2", 10)
        momentum = g2 - g1
        if momentum > 0:
            momentum_str = f"+{momentum} marks (Improved in Exam 2)"
        elif momentum < 0:
            momentum_str = f"{momentum} marks (Dropped in Exam 2)"
        else:
            momentum_str = "0 marks (Same as Exam 1)"
            
        return {
            "prediction": prediction,
            "status": "PASS" if prediction == 1 else "FAIL / AT-RISK",
            "pass_probability": pass_prob,
            "fail_probability": fail_prob,
            "risk_level": risk_level,
            "risk_badge": risk_badge,
            "momentum_str": momentum_str,
            "momentum_val": momentum
        }

    def simulate_study_time_impact(self, base_student: dict) -> pd.DataFrame:
        """
        Simulates how passing probability changes across weekly study time levels (1 to 4).
        """
        labels = {
            1: "Less than 2 hrs/week",
            2: "2 to 5 hrs/week",
            3: "5 to 10 hrs/week",
            4: "More than 10 hrs/week"
        }
        results = []
        for s_level in [1, 2, 3, 4]:
            temp = dict(base_student)
            temp["studytime"] = s_level
            sim_df = pd.DataFrame([temp])[self.features]
            p = round(float(self.model.predict_proba(sim_df)[0][self.pass_index]) * 100, 1)
            results.append({
                "Study_Time_Level": labels[s_level],
                "Pass_Probability": p
            })
        return pd.DataFrame(results)

    def simulate_absence_impact(self, base_student: dict, max_absences=26) -> pd.DataFrame:
        """
        Simulates how passing probability changes as school absences increase.
        """
        results = []
        for a_level in range(0, max_absences, 2):
            temp = dict(base_student)
            temp["absences"] = a_level
            sim_df = pd.DataFrame([temp])[self.features]
            p = round(float(self.model.predict_proba(sim_df)[0][self.pass_index]) * 100, 1)
            results.append({
                "Absences": a_level,
                "Pass_Probability": p
            })
        return pd.DataFrame(results)
