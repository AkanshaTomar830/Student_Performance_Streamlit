# Student Performance Prediction & Risk Dashboard

An AI-powered academic decision-support dashboard built using **Streamlit** and a **Random Forest Classifier** trained on the UCI Student Performance Dataset.

The system helps educators, counselors, and students evaluate academic progress early in the term, understand key drivers of success, and simulate how improved study habits or attendance impact passing likelihood.

---

## Features

- **Individual Student Assessment:** Predicts final examination outcome (Pass vs. Fail) using continuous grades ($G_1, G_2$), past failures, attendance, and study hours.
- **Explainable Insights (XAI):** Visualizes empirical feature importances directly from the trained Random Forest model.
- **What-If Sensitivity Simulation:** Interactively simulates how shifting weekly study time or reducing absences changes passing probabilities in real-time.
- **Built-in Student Profiles:** Includes quick-load example personas (*Top Student*, *Average Student*, *Struggling Student*, *Comeback Student*, *Attendance Risk Student*) for demonstrations.
- **Downloadable Reports:** Generates structured evaluation summaries (.txt) with one click.

---

## Dataset

Trained on the **UCI Student Performance Data Set** (Cortez & Silva, 2008), which records academic reports and survey attributes of secondary school students in Portugal.

Key predictors utilized:
- `G2`: Second period continuous evaluation (0–20)
- `G1`: First period continuous evaluation (0–20)
- `failures`: Number of past class failures (0–4)
- `absences`: Total days absent from school (0–93)
- `studytime`: Weekly study commitment (1: <2 hrs, 2: 2–5 hrs, 3: 5–10 hrs, 4: >10 hrs)

Classification cutoff: Final Grade $G_3 \ge 10$ is Pass ($1$), and $G_3 < 10$ is Fail ($0$).

---

## Project Structure

```text
Student_Performance_Streamlit/
├── .vscode/
│   └── launch.json                      # VS Code run & debug config
├── Models/
│   ├── app_features.pkl                 # Feature names vector
│   └── student_prediction_app_model.pkl # Trained Random Forest model
├── .gitignore                           # Git ignore rules
├── app.py                               # Main Streamlit application
├── README.md                            # Project documentation
└── requirements.txt                     # Python package dependencies
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/AkanshaTomar830/Student_Performance_Streamlit.git
cd Student_Performance_Streamlit
```

### 2. Create and Activate a Virtual Environment (Optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
Or:
```bash
python -m streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## License

This project is developed for educational and academic Project-Based Learning (PBL) demonstration purposes.
