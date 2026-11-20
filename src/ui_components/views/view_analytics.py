"""
MEMBER 3: View - Dataset Analytics & Visualizations (Integrating Member 1's Work)
"""

import streamlit as st
import pandas as pd
from src.data_analytics import (
    load_student_dataset,
    get_dataset_overview,
    create_pass_fail_donut,
    create_grade_distribution_chart,
    create_studytime_impact_chart,
    create_failures_impact_chart,
    create_correlation_heatmap
)


def render_analytics_view():
    """
    Renders Tab 2: Dataset Exploration & Exploratory Data Analysis (EDA).
    Owned and developed by Member 1.
    """
    st.markdown("### Dataset Exploration & Visualizations")
    st.caption("Exploratory Data Analysis (EDA) on the UCI Student Performance benchmark dataset.")

    try:
        df = load_student_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    summary = get_dataset_overview(df)

    # High-level cohort KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Students Analyzed", summary["total_students"])
    with k2:
        st.metric("Cohort Pass Rate", f"{summary['pass_rate']}%", f"{summary['pass_count']} Passed")
    with k3:
        st.metric("Average Final Exam (G3)", f"{summary['avg_g3']} / 20")
    with k4:
        st.metric("Average Absences", f"{summary['avg_absences']} days")

    st.markdown("<br>", unsafe_allow_html=True)

    # First row of charts: Pass/Fail Ratio & Grade Distribution
    col_chart1, col_chart2 = st.columns([1, 2], gap="medium")
    
    with col_chart1:
        st.markdown("#### Overall Pass vs. Fail Ratio")
        donut = create_pass_fail_donut(df)
        st.altair_chart(donut, use_container_width=True)

    with col_chart2:
        st.markdown("#### Score Distribution across Exam Periods")
        grade_dist = create_grade_distribution_chart(df)
        st.altair_chart(grade_dist, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Second row of charts: Behavioral drivers
    col_chart3, col_chart4 = st.columns(2, gap="medium")
    
    with col_chart3:
        st.markdown("#### Impact of Weekly Study Time on Pass Rate")
        study_chart = create_studytime_impact_chart(df)
        st.altair_chart(study_chart, use_container_width=True)
        st.caption("Students studying > 10 hours/week consistently achieve higher pass rates.")

    with col_chart4:
        st.markdown("#### Impact of Past Course Failures on Pass Rate")
        fail_chart = create_failures_impact_chart(df)
        st.altair_chart(fail_chart, use_container_width=True)
        st.caption("Past failures create a sharp drop in probability of passing future exams.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Correlation Heatmap
    st.markdown("#### Correlation Heatmap (Academic & Behavioral Factors)")
    corr_chart = create_correlation_heatmap(df)
    st.altair_chart(corr_chart, use_container_width=True)
    st.caption("Exam 1 (G1) and Exam 2 (G2) exhibit strong positive correlation (> 0.85) with final performance (G3).")

    # Data Table Preview with Interactive Filtering
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Explore Raw Dataset Sample (Filter & Search)"):
        status_filter = st.multiselect("Filter by Result Status", ["Pass", "Fail"], default=["Pass", "Fail"])
        filtered_df = df[df["status"].isin(status_filter)]
        st.dataframe(filtered_df[["studytime", "failures", "absences", "G1", "G2", "G3", "status"]], use_container_width=True)
        st.caption(f"Showing {len(filtered_df)} of {len(df)} student records.")
