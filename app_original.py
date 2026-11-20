
import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    model = joblib.load("Models/student_prediction_app_model.pkl")
    features = joblib.load("Models/app_features.pkl")
    return model, features

model, features = load_model()

# Application heading
st.title(" Student Performance Prediction")
st.write(
)

st.divider()

st.subheader("Enter Student Details")

# Student inputs
col1, col2 = st.columns(2)

with col1:
    studytime = st.selectbox(
        "Study Time",
        [1, 2, 3, 4],
        index=1
    )

    failures = st.number_input(
        "Previous Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences = st.number_input(
        "Absences",
        min_value=0,
        max_value=100,
        value=4
    )

with col2:
    g1 = st.slider("First Period Grade (G1)", 0, 20, 10)
    g2 = st.slider("Second Period Grade (G2)", 0, 20, 12)

# Prediction button
if st.button("Predict Student Result", use_container_width=True):

    student = pd.DataFrame([{
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": g1,
        "G2": g2
    }])

    student = student[features]

    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]

    pass_index = list(model.classes_).index(1)
    fail_index = list(model.classes_).index(0)

    pass_probability = probabilities[pass_index] * 100
    fail_probability = probabilities[fail_index] * 100

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Predicted Result: PASS")
    else:
        st.error("Predicted Result: FAIL")

    st.write(f"Pass Probability: {pass_probability:.2f}%")
    st.write(f"Fail Probability: {fail_probability:.2f}%")

    st.caption(
        "This is an educational prediction, not a guaranteed result."
    )
