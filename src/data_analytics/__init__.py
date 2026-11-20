"""
Data Analytics Package (Member 1)
"""
from .loader import load_student_dataset, get_dataset_overview
from .preprocessing import prepare_features_and_target, calculate_correlations
from .visualizations import (
    create_pass_fail_donut,
    create_grade_distribution_chart,
    create_studytime_impact_chart,
    create_failures_impact_chart,
    create_correlation_heatmap
)

__all__ = [
    "load_student_dataset",
    "get_dataset_overview",
    "prepare_features_and_target",
    "calculate_correlations",
    "create_pass_fail_donut",
    "create_grade_distribution_chart",
    "create_studytime_impact_chart",
    "create_failures_impact_chart",
    "create_correlation_heatmap"
]
