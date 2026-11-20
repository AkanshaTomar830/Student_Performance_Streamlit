"""
MEMBER 3: View - What-If Sensitivity Simulator
"""

import streamlit as st
import altair as alt


def render_simulation_view(predictor, current_student: dict):
    """
    Renders Tab 4: What-If Sensitivity Simulator.
    Shows real-time impact of altering weekly study hours or attendance.
    """
    st.markdown("### What-If Sensitivity Simulator")
    st.write(
        "Demonstrate how modifying student habits—such as increasing dedicated study hours or "
        "improving school attendance—dynamically shifts the student's passing likelihood."
    )

    st.markdown("#### Scenario 1: Impact of Increasing Weekly Study Hours")
    st.caption(f"Holding Exam 1 ({current_student['G1']}), Exam 2 ({current_student['G2']}), and Absences ({current_student['absences']}) constant:")

    study_sim_df = predictor.simulate_study_time_impact(current_student)

    study_chart = alt.Chart(study_sim_df).mark_bar(color="#3b82f6", cornerRadius=6).encode(
        x=alt.X("Pass_Probability:Q", title="Chance of Passing (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Study_Time_Level:N", sort=None, title=""),
        tooltip=["Study_Time_Level", alt.Tooltip("Pass_Probability:Q", title="Pass Chance (%)", format=".1f")]
    ).properties(height=180)

    st.altair_chart(study_chart, use_container_width=True)

    st.markdown("#### Scenario 2: Attendance Decay Curve (0 to 24 Days Absent)")
    st.caption("See how the probability of passing degrades as absences accumulate:")

    abs_sim_df = predictor.simulate_absence_impact(current_student, max_absences=26)

    line_chart = alt.Chart(abs_sim_df).mark_line(point=True, color="#8b5cf6", strokeWidth=3).encode(
        x=alt.X("Absences:Q", title="Total Days Absent"),
        y=alt.Y("Pass_Probability:Q", scale=alt.Scale(domain=[0, 100]), title="Chance of Passing (%)"),
        tooltip=["Absences", alt.Tooltip("Pass_Probability:Q", title="Pass Chance (%)", format=".1f")]
    ).properties(height=260)

    st.altair_chart(line_chart, use_container_width=True)
    st.caption("Notice: When absences exceed critical thresholds (around 10+ days), passing probability drops sharply.")
