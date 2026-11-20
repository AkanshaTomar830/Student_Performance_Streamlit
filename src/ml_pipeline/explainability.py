"""
MEMBER 2: Model Explainability (XAI) Module
Extracts feature importances and builds explainability visualizations.
"""

import pandas as pd
import altair as alt
from typing import List


FEATURE_LABEL_MAP = {
    "G2": "Second Exam Marks (G2)",
    "G1": "First Exam Marks (G1)",
    "failures": "Past Course Failures",
    "absences": "Days Absent from School",
    "studytime": "Weekly Study Time"
}


def get_feature_importance_df(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extracts empirical feature importances from the trained tree ensemble.
    """
    importances = model.feature_importances_ * 100
    df_fi = pd.DataFrame({
        "Feature_Code": feature_names,
        "Factor": [FEATURE_LABEL_MAP.get(f, f) for f in feature_names],
        "Importance_Pct": [round(val, 2) for val in importances]
    }).sort_values(by="Importance_Pct", ascending=False)
    
    return df_fi


def create_feature_importance_chart(df_fi: pd.DataFrame) -> alt.Chart:
    """
    Generates an interactive horizontal bar chart for feature importances.
    """
    chart = alt.Chart(df_fi).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
        x=alt.X("Importance_Pct:Q", title="Relative Impact on Prediction (%)"),
        y=alt.Y("Factor:N", sort="-x", title=""),
        color=alt.Color("Importance_Pct:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
        tooltip=["Factor", alt.Tooltip("Importance_Pct:Q", title="Importance (%)", format=".2f")]
    ).properties(height=240)
    
    return chart
