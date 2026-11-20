import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from datetime import datetime

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI | Student Performance Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (CSS for Clean, Modern Academic Look)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Font & Header Tweaks */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        border-radius: 16px;
        padding: 24px 30px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #cbd5e1;
        margin-top: 6px;
        margin-bottom: 14px;
        line-height: 1.5;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        background: rgba(255, 255, 255, 0.15);
        color: #e2e8f0;
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .badge-accent {
        background: rgba(20, 184, 166, 0.35);
        color: #5eead4;
        border-color: rgba(94, 234, 212, 0.4);
    }

    /* Card Containers */
    .custom-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
    }
    @media (prefers-color-scheme: dark) {
        .custom-card {
            background-color: #1e293b;
            border-color: #334155;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
    }
    
    .card-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    @media (prefers-color-scheme: dark) {
        .card-header {
            color: #f8fafc;
        }
    }

    /* Metric Boxes */
    .metric-pill {
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
    }
    @media (prefers-color-scheme: dark) {
        .metric-pill {
            background: #0f172a;
            border-color: #334155;
        }
    }
    
    .status-pass {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 16px;
        color: #065f46;
    }
    .status-fail {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 14px;
        padding: 16px;
        color: #991b1b;
    }
    @media (prefers-color-scheme: dark) {
        .status-pass {
            color: #6ee7b7;
        }
        .status-fail {
            color: #fca5a5;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Features
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Models/student_prediction_app_model.pkl")
        features = joblib.load("Models/app_features.pkl")
        return model, features, None
    except Exception as e:
        return None, None, str(e)

model, features, load_err = load_model()

if load_err:
    st.error(f" Failed to load ML model: {load_err}")
    st.stop()

# Human-readable labels for Study Time
STUDY_TIME_MAP = {
    1: "1: < 2 Hours / week",
    2: "2: 2 – 5 Hours / week",
    3: "3: 5 – 10 Hours / week",
    4: "4: > 10 Hours / week"
}

# ---------------------------------------------------------
# Sidebar: Presets, Metadata & Project Details
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("##  **EduPredict AI**")
    st.caption("Machine Learning Academic Early Warning System")
    st.divider()

    st.markdown("###  Quick-Load Student Persona")
    st.caption("Use preset personas to quickly demonstrate different prediction scenarios:")
    persona = st.selectbox(
        "Select Persona Demo",
        ["Custom Input", " High Achiever (Safe)", " Borderline Student (Watchlist)", " At-Risk Student (Critical Alert)"],
        index=0
    )

    # Persona Presets
    if persona == " High Achiever (Safe)":
        default_study = 3
        default_fail = 0
        default_abs = 2
        default_g1 = 16
        default_g2 = 17
        default_name = "Maya Lin"
    elif persona == " Borderline Student (Watchlist)":
        default_study = 2
        default_fail = 1
        default_abs = 8
        default_g1 = 10
        default_g2 = 10
        default_name = "Jordan Blake"
    elif persona == " At-Risk Student (Critical Alert)":
        default_study = 1
        default_fail = 2
        default_abs = 18
        default_g1 = 6
        default_g2 = 7
        default_name = "Sam Taylor"
    else:
        default_study = 2
        default_fail = 0
        default_abs = 4
        default_g1 = 11
        default_g2 = 12
        default_name = "Alex Morgan"

    st.divider()
    st.markdown("###  Student Information")
    student_name = st.text_input("Student Name", value=default_name)
    student_id = st.text_input("Student / Roll ID", value="STU-2026-089")
    course_name = st.selectbox("Subject Track", ["Mathematics (Core)", "Language / Portuguese", "Applied Sciences"])

    st.divider()
    st.markdown("### ℹ Project & Model Specs")
    st.markdown("""
    - **Architecture:** Random Forest Classifier
    - **Classification Target:** Pass (G3 ≥ 10) vs Fail (G3 < 10)
    - **Grading Scale:** Portuguese 0–20 standard
    - **Benchmark Dataset:** UCI Student Performance
    - **PBL Project Team:** Academic Analytics Lab
    """)

# ---------------------------------------------------------
# Hero Banner Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span> Student Academic Performance & Risk Dashboard</span>
    </div>
    <div class="hero-subtitle" style="margin-bottom: 0;">
        An Explainable AI (XAI) early warning system designed for educators and academic advisors to identify at-risk students, simulate behavioral interventions, and improve retention.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab_eval, tab_sim, tab_xai = st.tabs([
    " Individual Assessment",
    " What-If Sensitivity Lab",
    " Model Intelligence & Insights"
])

# =========================================================
# TAB 1: INDIVIDUAL ASSESSMENT
# =========================================================
with tab_eval:
    st.markdown("###  Student Input Profile")
    st.caption("Provide the student's academic history, current term assessments, and study habits to generate an assessment.")

    col_in1, col_in2 = st.columns(2, gap="large")

    with col_in1:
        st.markdown("""
        <div style="font-weight: 600; font-size: 1.05rem; margin-bottom: 8px;">
             Academic Track Record
        </div>
        """, unsafe_allow_html=True)

        g1 = st.slider(
            "Period 1 Assessment (G1)",
            min_value=0,
            max_value=20,
            value=default_g1,
            help="First period continuous assessment score (Scale 0-20. 10 is the passing cutoff)."
        )
        st.caption(f"Score status: **{g1}/20** ({g1*5}%) — {'🟢 Passing' if g1 >= 10 else '🔴 At-Risk'}")

        g2 = st.slider(
            "Period 2 Assessment (G2)",
            min_value=0,
            max_value=20,
            value=default_g2,
            help="Second period continuous assessment score (Scale 0-20. 10 is the passing cutoff)."
        )
        st.caption(f"Score status: **{g2}/20** ({g2*5}%) — {'🟢 Passing' if g2 >= 10 else '🔴 At-Risk'}")

        failures = st.selectbox(
            "Past Course Failures",
            options=[0, 1, 2, 3, 4],
            index=default_fail,
            help="Number of past academic class failures (0 to 4+)."
        )

    with col_in2:
        st.markdown("""
        <div style="font-weight: 600; font-size: 1.05rem; margin-bottom: 8px;">
             Study Habits & Attendance
        </div>
        """, unsafe_allow_html=True)

        studytime_key = st.selectbox(
            "Weekly Study Commitment",
            options=[1, 2, 3, 4],
            format_func=lambda x: STUDY_TIME_MAP[x],
            index=default_study - 1,
            help="Dedicated independent study hours outside classroom lectures."
        )

        absences = st.number_input(
            "Recorded School Absences",
            min_value=0,
            max_value=93,
            value=default_abs,
            step=1,
            help="Total sessions missed. Dataset median is ~4. More than 10 absences significantly correlates with academic distress."
        )
        if absences > 10:
            st.caption(" **High Absence Alert:** Attendance is a key leading indicator of course dropout.")
        else:
            st.caption(f"Attendance status: **{absences} absences** (Within acceptable attendance margin).")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button(" Run Comprehensive Performance Evaluation", use_container_width=True, type="primary")

    # Perform Prediction
    student_df = pd.DataFrame([{
        "studytime": studytime_key,
        "failures": failures,
        "absences": absences,
        "G1": g1,
        "G2": g2
    }])[features]

    prediction = model.predict(student_df)[0]
    probabilities = model.predict_proba(student_df)[0]

    pass_index = list(model.classes_).index(1)
    fail_index = list(model.classes_).index(0)

    pass_prob = probabilities[pass_index] * 100
    fail_prob = probabilities[fail_index] * 100

    # Risk Tiering
    if pass_prob >= 75:
        risk_level = "Low Risk (On Track)"
        risk_badge = "🟢 Safe"
        risk_color = "#10b981"
    elif pass_prob >= 50:
        risk_level = "Moderate Risk (Watchlist)"
        risk_badge = "🟡 Watchlist"
        risk_color = "#f59e0b"
    else:
        risk_level = "High Critical Risk (Intervention Needed)"
        risk_badge = "🔴 Critical"
        risk_color = "#ef4444"

    momentum = g2 - g1
    if momentum > 0:
        momentum_str = f"+{momentum} pts (Improving )"
    elif momentum < 0:
        momentum_str = f"{momentum} pts (Declining )"
    else:
        momentum_str = "0 pts (Consistent )"

    st.divider()
    st.markdown(f"###  Evaluation Results: **{student_name}** (`{student_id}`)")

    # 4 Key Metric Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(
            label="Predicted Final Status",
            value="PASS" if prediction == 1 else "FAIL / AT-RISK",
            delta="Qualified" if prediction == 1 else "- Action Required",
            delta_color="normal" if prediction == 1 else "inverse"
        )
    with m_col2:
        st.metric(
            label="Pass Confidence",
            value=f"{pass_prob:.1f}%",
            delta=f"{pass_prob - 50:+.1f}% vs Cutoff"
        )
    with m_col3:
        st.metric(
            label="Academic Risk Tier",
            value=risk_badge,
            help=f"Classified under: {risk_level}"
        )
    with m_col4:
        st.metric(
            label="Grade Momentum (G2 - G1)",
            value=momentum_str,
            help="Trajectory difference between Period 1 and Period 2 continuous assessments."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Probability Bar
    st.markdown("####  Outcome Probability Distribution")
    prob_chart_df = pd.DataFrame({
        "Outcome": ["Pass Probability", "Fail Risk"],
        "Probability (%)": [pass_prob, fail_prob],
        "Color": ["#10b981", "#ef4444"]
    })

    bar_chart = alt.Chart(prob_chart_df).mark_bar(cornerRadius=6, size=24).encode(
        x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100]), title="Model Probability (%)"),
        y=alt.Y("Outcome:N", title="", sort="-x"),
        color=alt.Color("Color:N", scale=None),
        tooltip=["Outcome", alt.Tooltip("Probability (%):Q", format=".1f")]
    ).properties(height=140)

    st.altair_chart(bar_chart, use_container_width=True)

    # Downloadable Student Assessment Summary Report
    st.markdown("<br>", unsafe_allow_html=True)
    report_text = f"""==================================================
EDUPREDICT AI - STUDENT ACADEMIC ASSESSMENT REPORT
Date of Report: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================

STUDENT PROFILE:
- Name: {student_name}
- Student ID: {student_id}
- Subject Track: {course_name}

INPUT METRICS:
- First Period Grade (G1): {g1} / 20
- Second Period Grade (G2): {g2} / 20
- Past Failures: {failures}
- Weekly Study Time: {STUDY_TIME_MAP[studytime_key]}
- Absences: {absences}

PREDICTIVE DIAGNOSTICS:
- Predicted Outcome: {'PASS' if prediction == 1 else 'FAIL / AT-RISK'}
- Pass Probability: {pass_prob:.2f}%
- Failure Risk Probability: {fail_prob:.2f}%
- Risk Classification: {risk_level}
- Grade Trajectory: {momentum_str}

==================================================
Notice: This is an ML diagnostic tool for educational planning and does not represent an absolute guarantee.
"""
    st.download_button(
        label=" Download Formal Student Evaluation Report (.txt)",
        data=report_text,
        file_name=f"Assessment_{student_id}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

# =========================================================
# TAB 2: WHAT-IF SENSITIVITY LAB
# =========================================================
with tab_sim:
    st.markdown("###  What-If Sensitivity Simulator")
    st.write(
        "Demonstrate the practical utility of this ML model in Project-Based Learning: "
        "explore how modifying student habits (such as increasing study hours or reducing absences) "
        "can dynamically shift passing probability."
    )

    st.markdown("#### Scenario: Varying Study Hours (Holding Grades & Absences Fixed)")
    study_sim_data = []
    for s_level in [1, 2, 3, 4]:
        sim_df = pd.DataFrame([{
            "studytime": s_level,
            "failures": failures,
            "absences": absences,
            "G1": g1,
            "G2": g2
        }])[features]
        p_val = model.predict_proba(sim_df)[0][pass_index] * 100
        study_sim_data.append({
            "Study Time Level": STUDY_TIME_MAP[s_level],
            "Pass Probability (%)": round(p_val, 1)
        })
    df_study_sim = pd.DataFrame(study_sim_data)

    study_chart = alt.Chart(df_study_sim).mark_bar(color="#3b82f6", cornerRadius=6).encode(
        x=alt.X("Pass Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Study Time Level:N", sort=None, title=""),
        tooltip=["Study Time Level", "Pass Probability (%)"]
    ).properties(height=180)
    st.altair_chart(study_chart, use_container_width=True)

    st.markdown("#### Scenario: Attendance Sensitivity Curve (Impact of 0 to 25 Absences)")
    absence_sim_data = []
    for a_level in range(0, 26, 2):
        sim_df = pd.DataFrame([{
            "studytime": studytime_key,
            "failures": failures,
            "absences": a_level,
            "G1": g1,
            "G2": g2
        }])[features]
        p_val = model.predict_proba(sim_df)[0][pass_index] * 100
        absence_sim_data.append({
            "Absences": a_level,
            "Pass Probability (%)": round(p_val, 1)
        })
    df_abs_sim = pd.DataFrame(absence_sim_data)

    line_chart = alt.Chart(df_abs_sim).mark_line(point=True, color="#8b5cf6", strokeWidth=3).encode(
        x=alt.X("Absences:Q", title="Total Recorded Absences"),
        y=alt.Y("Pass Probability (%):Q", scale=alt.Scale(domain=[0, 100]), title="Passing Probability (%)"),
        tooltip=["Absences", "Pass Probability (%)"]
    ).properties(height=260)
    st.altair_chart(line_chart, use_container_width=True)
    st.caption("Notice how attendance drops trigger non-linear decay in success probability once absences exceed critical thresholds.")

# =========================================================
# TAB 3: MODEL INTELLIGENCE & XAI
# =========================================================
with tab_xai:
    st.markdown("###  Model Explainability & Feature Importance")
    st.write(
        "For academic evaluations and PBL defenses, it is crucial to explain **why** the model makes its decisions. "
        "Below are the empirical feature importances computed by the Random Forest Classifier across all ensemble decision trees."
    )

    feat_names_readable = {
        "G2": "Period 2 Grade (G2)",
        "G1": "Period 1 Grade (G1)",
        "failures": "Past Course Failures",
        "absences": "Number of Absences",
        "studytime": "Weekly Study Time"
    }

    importances = model.feature_importances_ * 100
    fi_df = pd.DataFrame({
        "Feature": [feat_names_readable.get(f, f) for f in features],
        "Importance (%)": [round(val, 2) for val in importances]
    }).sort_values(by="Importance (%)", ascending=False)

    fi_chart = alt.Chart(fi_df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
        x=alt.X("Importance (%):Q", title="Relative Predictive Importance (%)"),
        y=alt.Y("Feature:N", sort="-x", title=""),
        color=alt.Color("Importance (%):Q", scale=alt.Scale(scheme="tealblues"), legend=None),
        tooltip=["Feature", "Importance (%)"]
    ).properties(height=250)

    st.altair_chart(fi_chart, use_container_width=True)

    col_xai1, col_xai2 = st.columns(2)
    with col_xai1:
        st.markdown("####  Key Takeaways")
        st.markdown("""
        1. **Second Period Grade ($G2$) is the Dominant Factor (~58%):**
           Recent academic performance strongly correlates with final examination outcome ($G3$).
        2. **First Period Grade ($G1$) provides Baseline Foundation (~29%):**
           Combined, continuous assessments ($G1 + G2$) account for over **87%** of predictive power.
        3. **Behavioral Variables Act as Mediators (~13%):**
           Past failures, absences, and study time act as critical risk modulators, especially for borderline students near the 10-point threshold.
        """)
    with col_xai2:
        st.markdown("####  Model Architecture Specifications")
        st.markdown(f"""
        - **Algorithm:** `{type(model).__name__}`
        - **Number of Estimators (Trees):** `{getattr(model, 'n_estimators', 100)}`
        - **Splitting Criterion:** `{getattr(model, 'criterion', 'gini').capitalize()}`
        - **Input Dimensionality:** `{len(features)} Features`
        - **Decision Boundary:** Cutoff at P(Pass) ≥ 0.50
        - **Target Attribute:** Final Grade G3 ≥ 10 (Binary encoded: 0 = Fail, 1 = Pass)
        """)



# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    " **EduPredict AI Dashboard** · Built for Project-Based Learning (PBL) Academic Demonstration · "
    "Random Forest Machine Learning Classifier on UCI Student Performance Dataset."
)
