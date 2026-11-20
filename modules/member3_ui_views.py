"""
MEMBER 3: STREAMLIT DASHBOARD DESIGN, UI & INTEGRATION
This file contains all the work for Member 3:
- Custom styling and CSS
- Sidebar controls, student personas, and team roles
- Tab 1: Live Student Predictor and report generator
- Tab 4: What-If Habit and Attendance Sensitivity Simulator
"""

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from modules.member2_ml_models import predict_student, simulate_study_time, simulate_absences

STUDY_TIME_MAP = {
    1: "Less than 2 hours a week",
    2: "2 to 5 hours a week",
    3: "5 to 10 hours a week",
    4: "More than 10 hours a week"
}


def apply_styles():
    """Injects clean CSS styling."""
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        .hero-container {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
            border-radius: 14px;
            padding: 20px 26px;
            color: #ffffff;
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            color: #cbd5e1;
            margin-top: 6px;
            line-height: 1.5;
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renders the sidebar navigation, student presets, and team roles."""
    with st.sidebar:
        st.markdown("## **Student Pass Predictor**")
        st.caption("AI-Powered Academic Decision-Support System")
        st.divider()

        st.markdown("### Try Example Students")
        st.caption("Pick an example student to pre-fill the form:")
        persona = st.selectbox(
            "Choose an Example",
            [
                "Custom Input",
                "Top Student (Safe)",
                "Average Student (Borderline)",
                "Struggling Student (At Risk)",
                "Comeback Student (Improving)",
                "Attendance Risk Student"
            ],
            index=0
        )

        if persona == "Top Student (Safe)":
            default_study = 3; default_fail = 0; default_abs = 2; default_g1 = 16; default_g2 = 17; default_name = "Maya Lin"
        elif persona == "Average Student (Borderline)":
            default_study = 2; default_fail = 1; default_abs = 8; default_g1 = 10; default_g2 = 10; default_name = "Jordan Blake"
        elif persona == "Struggling Student (At Risk)":
            default_study = 1; default_fail = 2; default_abs = 18; default_g1 = 6; default_g2 = 7; default_name = "Sam Taylor"
        elif persona == "Comeback Student (Improving)":
            default_study = 3; default_fail = 1; default_abs = 3; default_g1 = 8; default_g2 = 14; default_name = "David Miller"
        elif persona == "Attendance Risk Student":
            default_study = 2; default_fail = 0; default_abs = 20; default_g1 = 13; default_g2 = 11; default_name = "Chris Evans"
        else:
            default_study = 2; default_fail = 0; default_abs = 4; default_g1 = 11; default_g2 = 12; default_name = "Alex Morgan"

        st.divider()
        st.markdown("### Student Details")
        student_name = st.text_input("Student Name", value=default_name)
        student_id = st.text_input("Roll / ID Number", value="STU-2026-089")
        course_name = st.selectbox("Subject", ["Mathematics", "Language", "Applied Sciences"])

        st.divider()
        st.markdown("### Project Team & Roles")
        st.markdown("""
        - **Member 1:** Data Analysis, Preprocessing & EDA
        - **Member 2:** ML Models, Evaluation & Prediction
        - **Member 3:** Streamlit Dashboard & System Integration
        """)

    return {
        "student_name": student_name,
        "student_id": student_id,
        "course_name": course_name,
        "default_g1": default_g1,
        "default_g2": default_g2,
        "default_study": default_study,
        "default_fail": default_fail,
        "default_abs": default_abs
    }


def render_prediction_tab(sidebar_state):
    """Renders Tab 1: Live Student Predictor."""
    st.markdown("### Enter Student Information")
    st.caption("Fill in the academic marks and habits below to see the prediction.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Past & Current Exam Marks**")
        g1 = st.slider("First Exam Marks (G1)", 0, 20, sidebar_state["default_g1"], help="Scale 0 to 20. 10 is passing.")
        st.caption(f"Score: **{g1}/20** ({g1*5}%) — {'Passing' if g1 >= 10 else 'At-Risk (Below 10)'}")

        g2 = st.slider("Second Exam Marks (G2)", 0, 20, sidebar_state["default_g2"], help="Scale 0 to 20. 10 is passing.")
        st.caption(f"Score: **{g2}/20** ({g2*5}%) — {'Passing' if g2 >= 10 else 'At-Risk (Below 10)'}")

        failures = st.selectbox("Past Class Failures", [0, 1, 2, 3, 4], index=sidebar_state["default_fail"])

    with col2:
        st.markdown("**Study Time & Attendance**")
        studytime_key = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4],
            format_func=lambda x: STUDY_TIME_MAP[x],
            index=sidebar_state["default_study"] - 1
        )
        absences = st.number_input("Days Absent from School", 0, 93, sidebar_state["default_abs"], step=1)
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

    # Member 2's prediction function
    res = predict_student(student_data)

    st.divider()
    st.markdown(f"### Prediction Result for: **{sidebar_state['student_name']}** (`{sidebar_state['student_id']}`)")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Predicted Result",
            res["status"],
            delta="On Track" if res["prediction"] == 1 else "Needs Attention",
            delta_color="normal" if res["prediction"] == 1 else "inverse"
        )
    with m2:
        st.metric("Chance of Passing", f"{res['pass_prob']}%", delta=f"{res['pass_prob'] - 50:+.1f}% vs Passing Cutoff")
    with m3:
        st.metric("Risk Level", res["risk_badge"], help=f"Category: {res['risk_level']}")
    with m4:
        st.metric("Marks Improvement (G2 - G1)", res["momentum_str"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Chances of Passing vs. Failing")

    chart_df = pd.DataFrame({
        "Result": ["Chance of Passing", "Risk of Failing"],
        "Percentage (%)": [res["pass_prob"], res["fail_prob"]],
        "Color": ["#10b981", "#ef4444"]
    })
    bar_chart = alt.Chart(chart_df).mark_bar(cornerRadius=6, size=24).encode(
        x=alt.X("Percentage (%):Q", scale=alt.Scale(domain=[0, 100]), title="Percentage (%)"),
        y=alt.Y("Result:N", title="", sort="-x"),
        color=alt.Color("Color:N", scale=None),
        tooltip=["Result", alt.Tooltip("Percentage (%):Q", format=".1f")]
    ).properties(height=130)
    st.altair_chart(bar_chart, use_container_width=True)

    # Download Report
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

PREDICTION:
- Result: {res['status']}
- Chance of Passing: {res['pass_prob']}%
- Risk of Failing: {res['fail_prob']}%
- Risk Level: {res['risk_level']}
- Exam Progress: {res['momentum_str']}
==================================================
"""
    st.download_button(
        label="Download Student Report (.txt)",
        data=report_text,
        file_name=f"Report_{sidebar_state['student_id']}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

    return student_data


def render_simulation_tab(current_student):
    """Renders Tab 4: What-If Habit Sensitivity Simulator."""
    st.markdown("### What-If Habit Simulator")
    st.write("See how studying more hours or missing fewer days can improve a student's chance of passing.")

    st.markdown("#### 1. What Happens If Study Hours Increase?")
    st.caption(f"Holding Exam 1 ({current_student['G1']}), Exam 2 ({current_student['G2']}), and Absences ({current_student['absences']}) the same:")

    df_study = simulate_study_time(current_student)
    study_chart = alt.Chart(df_study).mark_bar(color="#3b82f6", cornerRadius=6).encode(
        x=alt.X("Pass_Chance:Q", title="Chance of Passing (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Study_Time:N", sort=None, title=""),
        tooltip=["Study_Time", "Pass_Chance"]
    ).properties(height=180)
    st.altair_chart(study_chart, use_container_width=True)

    st.markdown("#### 2. How Many Days Absent Can a Student Afford?")
    st.caption("See how passing probability drops as absences increase from 0 to 24 days:")

    df_abs = simulate_absences(current_student)
    line_chart = alt.Chart(df_abs).mark_line(point=True, color="#8b5cf6", strokeWidth=3).encode(
        x=alt.X("Absences:Q", title="Days Absent"),
        y=alt.Y("Pass_Chance:Q", scale=alt.Scale(domain=[0, 100]), title="Chance of Passing (%)"),
        tooltip=["Absences", "Pass_Chance"]
    ).properties(height=240)
    st.altair_chart(line_chart, use_container_width=True)
    st.caption("Notice: Missing too many classes quickly drops the chance of passing.")
