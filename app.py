import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from src.ml_pipeline import StudentPredictor
from src.ui_components import (
    apply_custom_styles,
    render_sidebar,
    render_prediction_view,
    render_analytics_view,
    render_models_view,
    render_simulation_view
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI | Student Performance Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS Styling (Member 3)
apply_custom_styles()

# ---------------------------------------------------------
# Initialize Prediction Engine (Member 2)
# ---------------------------------------------------------
@st.cache_resource
def get_predictor():
    return StudentPredictor()

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Failed to initialize machine learning model: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls & Team Attribution (Member 3)
# ---------------------------------------------------------
sidebar_state = render_sidebar()

# ---------------------------------------------------------
# Hero Banner Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        Student Academic Performance & Risk Dashboard
    </div>
    <div class="hero-subtitle">
        An AI-powered academic decision-support system to predict student outcomes, analyze cohort performance, benchmark machine learning models, and simulate habit interventions.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Multi-Module Tab Navigation
# ---------------------------------------------------------
tab_pred, tab_eda, tab_eval, tab_sim = st.tabs([
    "Student Pass Predictor",
    "Dataset Analytics & EDA",
    "Model Evaluation & XAI",
    "What-If Habit Simulator"
])

# Tab 1: Live Inference & Student Risk Scoring
with tab_pred:
    current_student = render_prediction_view(predictor, sidebar_state)

# Tab 2: Member 1's Dataset Analysis, Statistics & EDA
with tab_eda:
    render_analytics_view()

# Tab 3: Member 2's ML Model Evaluation, Confusion Matrix & Benchmarking
with tab_eval:
    render_models_view(predictor)

# Tab 4: Interactive What-If Habit & Attendance Sensitivity Simulator
with tab_sim:
    render_simulation_view(predictor, current_student)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "EduPredict AI Dashboard · Project-Based Learning (PBL) Collaboration · "
    "Modular architecture dividing Data Analytics (Member 1), ML Pipeline (Member 2), and Frontend UI (Member 3)."
)
