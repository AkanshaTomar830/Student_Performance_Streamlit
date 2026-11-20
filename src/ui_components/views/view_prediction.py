"""
MEMBER 3: View - Individual Student Predictor
Renders the primary assessment form, KPI cards, visual distribution, and report generation.
"""

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from typing import Dict, Any

STUDY_TIME_MAP = {
    1: "Less than 2 hours a week",
    2: "2 to 5 hours a week",
    3: "5 to 10 hours a week",
    4: "More than 10 hours a week"
}


def render_prediction_view(predictor, sidebar_state: Dict[str, Any]) -> dict:
    """
    Renders Tab 1: Live Student Predictor.
    Returns the current student input dictionary so other tabs can reference it.
    """
    st.markdown("### Enter Student Information")
    st.caption("Fill in the academic marks and habits below to run the prediction.")

    col_in1, col_in2 = st.columns(2, gap="large")

    with col_in1:
        st.markdown("**Past & Current Exam Marks**")

        g1 = st.slider(
            "First Exam Marks (G1)",
            min_value=0,
            max_value=20,
            value=sidebar_state["default_g1"],
            help="Score in the first exam (Scale is 0 to 20. 10 is passing)."
        )
        st.caption(f"Score: **{g1}/20** ({g1*5}%) — {'Passing' if g1 >= 10 else 'At-Risk (Below 10)'}")

        g2 = st.slider(
            "Second Exam Marks (G2)",
            min_value=0,
            max_value=20,
            value=sidebar_state["default_g2"],
            help="Score in the second exam (Scale is 0 to 20. 10 is passing)."
        )
        st.caption(f"Score: **{g2}/20** ({g2*5}%) — {'Passing' if g2 >= 10 else 'At-Risk (Below 10)'}")

        failures = st.selectbox(
            "Past Class Failures",
            options=[0, 1, 2, 3, 4],
            index=sidebar_state["default_fail"],
            help="How many subjects the student failed in previous terms."
        )

    with col_in2:
        st.markdown("**Study Time & Attendance**")

        studytime_key = st.selectbox(
            "Weekly Study Time",
            options=[1, 2, 3, 4],
            format_func=lambda x: STUDY_TIME_MAP[x],
            index=sidebar_state["default_study"] - 1,
            help="How many hours the student studies independently outside class."
        )

        absences = st.number_input(
            "Days Absent from School",
            min_value=0,
            max_value=93,
            value=sidebar_state["default_abs"],
            step=1,
            help="Total days missed. Most students miss around 4 days. Missing more than 10 days significantly increases risk."
        )
        if absences > 10:
            st.caption("Warning: High absences make it harder to pass.")
        else:
            st.caption(f"Attendance: **{absences} days absent** (Normal range).")

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Predict Student Result", use_container_width=True, type="primary")

    student_data = {
        "studytime": studytime_key,
        "failures": failures,
        "absences": absences,
        "G1": g1,
        "G2": g2
    }

    # Inference via Member 2's predictor
    res = predictor.predict_student(student_data)

    st.divider()
    st.markdown(f"### Prediction Result for: **{sidebar_state['student_name']}** (`{sidebar_state['student_id']}`)")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Predicted Result",
            value=res["status"],
            delta="On Track" if res["prediction"] == 1 else "Needs Attention",
            delta_color="normal" if res["prediction"] == 1 else "inverse"
        )
    with m2:
        st.metric(
            label="Chance of Passing",
            value=f"{res['pass_probability']}%",
            delta=f"{res['pass_probability'] - 50:+.1f}% vs Passing Cutoff"
        )
    with m3:
        st.metric(
            label="Risk Level",
            value=res["risk_badge"],
            help=f"Classified under: {res['risk_level']}"
        )
    with m4:
        st.metric(
            label="Marks Improvement (G2 - G1)",
            value=res["momentum_str"]
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Chances of Passing vs. Failing")
    
    chart_df = pd.DataFrame({
        "Result": ["Chance of Passing", "Risk of Failing"],
        "Percentage (%)": [res["pass_probability"], res["fail_probability"]],
        "Color": ["#10b981", "#ef4444"]
    })
    
    bar_chart = alt.Chart(chart_df).mark_bar(cornerRadius=6, size=24).encode(
        x=alt.X("Percentage (%):Q", scale=alt.Scale(domain=[0, 100]), title="Percentage (%)"),
        y=alt.Y("Result:N", title="", sort="-x"),
        color=alt.Color("Color:N", scale=None),
        tooltip=["Result", alt.Tooltip("Percentage (%):Q", format=".1f")]
    ).properties(height=140)
    
    st.altair_chart(bar_chart, use_container_width=True)

    # Downloadable summary
    st.markdown("<br>", unsafe_allow_html=True)
    report_text = f"""==================================================
STUDENT PERFORMANCE EVALUATION REPORT
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================

STUDENT DETAILS:
- Name: {sidebar_state['student_name']}
- Roll / ID: {sidebar_state['student_id']}
- Subject: {sidebar_state['course_name']}

INPUT VALUES:
- First Exam Marks (G1): {g1} / 20
- Second Exam Marks (G2): {g2} / 20
- Past Failures: {failures}
- Weekly Study Time: {STUDY_TIME_MAP[studytime_key]}
- Days Absent: {absences}

PREDICTIVE DIAGNOSTICS:
- Result: {res['status']}
- Chance of Passing: {res['pass_probability']}%
- Risk of Failing: {res['fail_probability']}%
- Risk Level: {res['risk_level']}
- Exam Progress: {res['momentum_str']}

==================================================
Notice: Developed for PBL Academic Early Warning Demonstration.
"""
    st.download_button(
        label="Download Student Report (.txt)",
        data=report_text,
        file_name=f"Report_{sidebar_state['student_id']}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

    return student_data
