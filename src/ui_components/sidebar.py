"""
MEMBER 3: Sidebar Controls & Project Metadata Module
"""

import streamlit as st
from typing import Dict, Any


def render_sidebar() -> Dict[str, Any]:
    """
    Renders the sidebar navigation, preset personas, student identifier inputs, and project team roles.
    Returns a dictionary of current sidebar state.
    """
    with st.sidebar:
        st.markdown("## **Student Pass Predictor**")
        st.caption("AI-Powered Academic Early Warning System")
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

        # Preset values
        if persona == "Top Student (Safe)":
            default_study = 3
            default_fail = 0
            default_abs = 2
            default_g1 = 16
            default_g2 = 17
            default_name = "Maya Lin"
        elif persona == "Average Student (Borderline)":
            default_study = 2
            default_fail = 1
            default_abs = 8
            default_g1 = 10
            default_g2 = 10
            default_name = "Jordan Blake"
        elif persona == "Struggling Student (At Risk)":
            default_study = 1
            default_fail = 2
            default_abs = 18
            default_g1 = 6
            default_g2 = 7
            default_name = "Sam Taylor"
        elif persona == "Comeback Student (Improving)":
            default_study = 3
            default_fail = 1
            default_abs = 3
            default_g1 = 8
            default_g2 = 14
            default_name = "David Miller"
        elif persona == "Attendance Risk Student":
            default_study = 2
            default_fail = 0
            default_abs = 20
            default_g1 = 13
            default_g2 = 11
            default_name = "Chris Evans"
        else:
            default_study = 2
            default_fail = 0
            default_abs = 4
            default_g1 = 11
            default_g2 = 12
            default_name = "Alex Morgan"

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
        
        st.caption("PBL Academic Project · UCI Student Performance Dataset")

    return {
        "persona": persona,
        "default_study": default_study,
        "default_fail": default_fail,
        "default_abs": default_abs,
        "default_g1": default_g1,
        "default_g2": default_g2,
        "student_name": student_name,
        "student_id": student_id,
        "course_name": course_name
    }
