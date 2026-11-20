"""
MEMBER 3: View - Model Evaluation & Comparison (Integrating Member 2's Work)
"""

import streamlit as st
import pandas as pd
from src.data_analytics import load_student_dataset, prepare_features_and_target
from src.ml_pipeline import (
    evaluate_primary_model,
    benchmark_model_comparison,
    get_feature_importance_df,
    create_feature_importance_chart
)


def render_models_view(predictor):
    """
    Renders Tab 3: Model Evaluation, Algorithm Comparison & Explainability (XAI).
    Owned and developed by Member 2.
    """
    st.markdown("### Machine Learning Model Evaluation & Insights")
    st.caption("Quantitative performance metrics, confusion matrix, algorithm comparisons, and feature importance.")

    try:
        df = load_student_dataset()
        X, y = prepare_features_and_target(df, predictor.features)
        metrics = evaluate_primary_model(predictor.model, X, y)
    except Exception as e:
        st.error(f"Error computing model evaluation: {e}")
        return

    # Top-line Evaluation KPIs
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("Model Accuracy", f"{metrics['accuracy']}%", "Primary Metric")
    with e2:
        st.metric("Precision (Pass)", f"{metrics['precision']}%", "Low False Positives")
    with e3:
        st.metric("Recall (Sensitivity)", f"{metrics['recall']}%", "Catches Failing Students")
    with e4:
        st.metric("F1 Score", f"{metrics['f1_score']}%", "Harmonic Balance")

    st.markdown("<br>", unsafe_allow_html=True)

    col_eval1, col_eval2 = st.columns(2, gap="large")

    with col_eval1:
        st.markdown("#### Confusion Matrix Breakdown")
        st.caption(f"Evaluated on {metrics['total_samples']} student benchmark records:")
        
        cm_df = pd.DataFrame({
            "Predicted: Fail": [metrics["true_negatives"], metrics["false_negatives"]],
            "Predicted: Pass": [metrics["false_positives"], metrics["true_positives"]]
        }, index=["Actual: Fail", "Actual: Pass"])
        
        st.table(cm_df)
        st.caption(
            f"**True Positives:** {metrics['true_positives']} | "
            f"**True Negatives:** {metrics['true_negatives']} | "
            f"**False Positives:** {metrics['false_positives']} | "
            f"**False Negatives:** {metrics['false_negatives']}"
        )

    with col_eval2:
        st.markdown("#### Algorithm Benchmark Comparison")
        st.caption("5-Fold Cross Validation comparison against baseline algorithms:")
        
        bench_df = benchmark_model_comparison(X, y, cv=5)
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
        st.caption("Random Forest provides superior generalization with the lowest variance across folds.")

    st.divider()

    # Feature Importance Section
    st.markdown("#### Feature Importance: What Factors Matter Most?")
    st.caption("Empirical feature importances extracted from the Random Forest tree ensemble:")

    df_fi = get_feature_importance_df(predictor.model, predictor.features)
    fi_chart = create_feature_importance_chart(df_fi)
    st.altair_chart(fi_chart, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Key Takeaways")
        st.markdown("""
        1. **Second Exam (G2) is the Dominant Factor (~58%):**
           Recent performance is the strongest signal of final course outcome.
        2. **First Exam (G1) provides Foundational Baseline (~29%):**
           Combined, continuous assessments (G1 + G2) account for over **87%** of predictive power.
        3. **Behavioral Features Decide Borderlines (~13%):**
           Past failures, attendance, and study hours determine outcomes for students close to the 10-point cutoff.
        """)
    with c2:
        st.markdown("##### Model Specifications")
        st.markdown(f"""
        - **Algorithm:** `{type(predictor.model).__name__}`
        - **Number of Estimators:** `{getattr(predictor.model, 'n_estimators', 100)} Decision Trees`
        - **Splitting Criterion:** `{getattr(predictor.model, 'criterion', 'gini').capitalize()}`
        - **Input Features:** `{len(predictor.features)} Selected Features`
        - **Decision Threshold:** `P(Pass) ≥ 0.50`
        """)
