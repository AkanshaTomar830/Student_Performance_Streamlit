"""
MEMBER 1: Visualizations Module
Creates interactive Altair charts for exploratory data analysis (EDA).
"""

import pandas as pd
import altair as alt


def create_pass_fail_donut(df: pd.DataFrame) -> alt.Chart:
    """
    Renders a donut chart of Pass vs Fail distribution in the dataset.
    """
    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    status_counts["Percentage"] = (status_counts["Count"] / status_counts["Count"].sum() * 100).round(1)
    
    color_scale = alt.Scale(domain=["Pass", "Fail"], range=["#10b981", "#ef4444"])
    
    chart = alt.Chart(status_counts).mark_arc(innerRadius=50, stroke="#fff").encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Status", type="nominal", scale=color_scale),
        tooltip=["Status", "Count", alt.Tooltip("Percentage:Q", format=".1f")]
    ).properties(height=240)
    
    return chart


def create_grade_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Compares the distributions of First Exam (G1), Second Exam (G2), and Final Exam (G3).
    """
    melted = df.melt(
        id_vars=["passed"],
        value_vars=["G1", "G2", "G3"],
        var_name="Exam",
        value_name="Score"
    )
    exam_labels = {"G1": "First Exam (G1)", "G2": "Second Exam (G2)", "G3": "Final Exam (G3)"}
    melted["Exam"] = melted["Exam"].map(exam_labels)
    
    chart = alt.Chart(melted).mark_bar(opacity=0.6, binSpacing=1).encode(
        x=alt.X("Score:Q", bin=alt.Bin(step=2), title="Exam Score (0-20)"),
        y=alt.Y("count()", title="Number of Students"),
        color=alt.Color("Exam:N", title="Exam Period", scale=alt.Scale(scheme="category10")),
        tooltip=["Exam:N", "count()"]
    ).properties(height=250)
    
    return chart


def create_studytime_impact_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Shows average final grade G3 and pass rate across 4 study time categories.
    """
    study_map = {
        1: "< 2 hrs/week",
        2: "2 - 5 hrs/week",
        3: "5 - 10 hrs/week",
        4: "> 10 hrs/week"
    }
    
    grouped = df.groupby("studytime").agg(
        avg_g3=("G3", "mean"),
        pass_rate=("passed", lambda x: (x.sum() / len(x)) * 100),
        count=("G3", "count")
    ).reset_index()
    
    grouped["Study_Hours"] = grouped["studytime"].map(study_map)
    grouped["avg_g3"] = grouped["avg_g3"].round(2)
    grouped["pass_rate"] = grouped["pass_rate"].round(1)
    
    chart = alt.Chart(grouped).mark_bar(cornerRadius=6, color="#3b82f6").encode(
        x=alt.X("Study_Hours:N", title="Weekly Study Commitment", sort=list(study_map.values())),
        y=alt.Y("pass_rate:Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=["Study_Hours", "pass_rate", "avg_g3", "count"]
    ).properties(height=240)
    
    return chart


def create_failures_impact_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Visualizes the steep drop in pass rate as past failures increase.
    """
    grouped = df.groupby("failures").agg(
        pass_rate=("passed", lambda x: (x.sum() / len(x)) * 100),
        student_count=("G3", "count")
    ).reset_index()
    
    grouped["pass_rate"] = grouped["pass_rate"].round(1)
    grouped["Failures_Label"] = grouped["failures"].astype(str) + " Failure(s)"
    
    chart = alt.Chart(grouped).mark_bar(cornerRadius=6).encode(
        x=alt.X("Failures_Label:N", title="Past Course Failures", sort=None),
        y=alt.Y("pass_rate:Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("pass_rate:Q", scale=alt.Scale(scheme="redyellowgreen"), legend=None),
        tooltip=["Failures_Label", "pass_rate", "student_count"]
    ).properties(height=240)
    
    return chart


def create_correlation_heatmap(df: pd.DataFrame) -> alt.Chart:
    """
    Computes and plots an Altair correlation heatmap of numeric academic variables.
    """
    cols = ["G1", "G2", "G3", "studytime", "failures", "absences", "age"]
    avail = [c for c in cols if c in df.columns]
    corr = df[avail].corr().round(2).reset_index().melt(id_vars="index")
    corr.columns = ["Variable 1", "Variable 2", "Correlation"]
    
    chart = alt.Chart(corr).mark_rect().encode(
        x=alt.X("Variable 1:N", title=""),
        y=alt.Y("Variable 2:N", title=""),
        color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
        tooltip=["Variable 1", "Variable 2", "Correlation"]
    ).properties(height=280)
    
    text = chart.mark_text().encode(
        text=alt.Text("Correlation:Q", format=".2f"),
        color=alt.condition("abs(datum.Correlation) > 0.5", alt.value("white"), alt.value("black"))
    )
    
    return chart + text
