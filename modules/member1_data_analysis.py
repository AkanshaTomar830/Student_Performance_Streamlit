"""
MEMBER 1: DATA ANALYSIS, PREPROCESSING & VISUALIZATIONS
This file contains all the work for Member 1:
- Loading and cleaning the student dataset
- Summary statistics and correlation analysis
- Interactive charts and exploratory data analysis (EDA)
"""

import os
import streamlit as st
import pandas as pd
import altair as alt

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")


@st.cache_data
def load_student_data():
    """Loads the UCI Mathematics student dataset."""
    df = pd.read_csv(os.path.abspath(DATA_PATH), sep=";")
    if "passed" not in df.columns and "G3" in df.columns:
        df["passed"] = (df["G3"] >= 10).astype(int)
        df["status"] = df["passed"].map({1: "Pass", 0: "Fail"})
    return df


def render_data_analysis_tab():
    """Renders the entire Data Analysis and Visualization tab in the dashboard."""
    st.markdown("### Dataset Analysis & Visualizations")
    st.caption("Exploratory Data Analysis (EDA) on the UCI Student Performance benchmark dataset.")

    try:
        df = load_student_data()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # 1. Dataset Overview Metrics
    total = len(df)
    pass_count = int((df["passed"] == 1).sum())
    pass_rate = round((pass_count / total) * 100, 1)
    avg_g3 = round(df["G3"].mean(), 2)
    avg_abs = round(df["absences"].mean(), 1)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Students", total)
    with k2:
        st.metric("Pass Rate", f"{pass_rate}%", f"{pass_count} Passed")
    with k3:
        st.metric("Average Final Mark (G3)", f"{avg_g3} / 20")
    with k4:
        st.metric("Average Days Absent", f"{avg_abs} days")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts: Pass/Fail Ratio & Grade Distribution
    col1, col2 = st.columns([1, 2], gap="medium")

    with col1:
        st.markdown("#### Pass vs. Fail Ratio")
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        status_counts["Percentage"] = (status_counts["Count"] / status_counts["Count"].sum() * 100).round(1)

        donut = alt.Chart(status_counts).mark_arc(innerRadius=50, stroke="#fff").encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Status:N", scale=alt.Scale(domain=["Pass", "Fail"], range=["#10b981", "#ef4444"])),
            tooltip=["Status", "Count", alt.Tooltip("Percentage:Q", format=".1f")]
        ).properties(height=240)
        st.altair_chart(donut, use_container_width=True)

    with col2:
        st.markdown("#### Marks Distribution across Exam Periods")
        melted = df.melt(id_vars=["passed"], value_vars=["G1", "G2", "G3"], var_name="Exam", value_name="Score")
        exam_labels = {"G1": "First Exam (G1)", "G2": "Second Exam (G2)", "G3": "Final Exam (G3)"}
        melted["Exam"] = melted["Exam"].map(exam_labels)

        grade_chart = alt.Chart(melted).mark_bar(opacity=0.6, binSpacing=1).encode(
            x=alt.X("Score:Q", bin=alt.Bin(step=2), title="Marks (0 to 20)"),
            y=alt.Y("count()", title="Number of Students"),
            color=alt.Color("Exam:N", title="Exam", scale=alt.Scale(scheme="category10")),
            tooltip=["Exam:N", "count()"]
        ).properties(height=240)
        st.altair_chart(grade_chart, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Behavioral Drivers
    col3, col4 = st.columns(2, gap="medium")

    with col3:
        st.markdown("#### Impact of Weekly Study Time on Pass Rate")
        study_map = {1: "< 2 hrs/week", 2: "2 - 5 hrs/week", 3: "5 - 10 hrs/week", 4: "> 10 hrs/week"}
        grouped_study = df.groupby("studytime").agg(
            pass_rate=("passed", lambda x: (x.sum() / len(x)) * 100),
            count=("G3", "count")
        ).reset_index()
        grouped_study["Study_Hours"] = grouped_study["studytime"].map(study_map)
        grouped_study["pass_rate"] = grouped_study["pass_rate"].round(1)

        study_chart = alt.Chart(grouped_study).mark_bar(cornerRadius=6, color="#3b82f6").encode(
            x=alt.X("Study_Hours:N", title="Study Time", sort=list(study_map.values())),
            y=alt.Y("pass_rate:Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Study_Hours", "pass_rate", "count"]
        ).properties(height=230)
        st.altair_chart(study_chart, use_container_width=True)

    with col4:
        st.markdown("#### Impact of Past Failures on Pass Rate")
        grouped_fail = df.groupby("failures").agg(
            pass_rate=("passed", lambda x: (x.sum() / len(x)) * 100),
            count=("G3", "count")
        ).reset_index()
        grouped_fail["pass_rate"] = grouped_fail["pass_rate"].round(1)
        grouped_fail["Label"] = grouped_fail["failures"].astype(str) + " Failure(s)"

        fail_chart = alt.Chart(grouped_fail).mark_bar(cornerRadius=6).encode(
            x=alt.X("Label:N", title="Past Failures", sort=None),
            y=alt.Y("pass_rate:Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("pass_rate:Q", scale=alt.Scale(scheme="redyellowgreen"), legend=None),
            tooltip=["Label", "pass_rate", "count"]
        ).properties(height=230)
        st.altair_chart(fail_chart, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Correlation Heatmap
    st.markdown("#### Correlation Heatmap (How Factors Relate to Each Other)")
    cols = ["G1", "G2", "G3", "studytime", "failures", "absences"]
    corr = df[cols].corr().round(2).reset_index().melt(id_vars="index")
    corr.columns = ["Factor 1", "Factor 2", "Correlation"]

    heat = alt.Chart(corr).mark_rect().encode(
        x=alt.X("Factor 1:N", title=""),
        y=alt.Y("Factor 2:N", title=""),
        color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
        tooltip=["Factor 1", "Factor 2", "Correlation"]
    ).properties(height=260)

    heat_text = heat.mark_text().encode(
        text=alt.Text("Correlation:Q", format=".2f"),
        color=alt.condition("abs(datum.Correlation) > 0.5", alt.value("white"), alt.value("black"))
    )
    st.altair_chart(heat + heat_text, use_container_width=True)

    # 5. Raw Data Preview
    with st.expander("Preview Raw Dataset Records"):
        status_filter = st.multiselect("Filter by Result", ["Pass", "Fail"], default=["Pass", "Fail"])
        sub_df = df[df["status"].isin(status_filter)]
        st.dataframe(sub_df[["studytime", "failures", "absences", "G1", "G2", "G3", "status"]], use_container_width=True)
