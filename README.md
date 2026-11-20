# Student Performance Prediction & Risk Dashboard

An AI-powered academic decision-support platform built using **Streamlit** and a **Random Forest Classifier** trained on the UCI Student Performance benchmark dataset.

The system predicts final exam outcomes (Pass vs. Fail), visualizes student data distributions, benchmarks machine learning algorithms, and simulates how study habits and attendance influence passing chances.

---

## 3-Member Clean Modular Structure

To make collaboration simple and organized, the code is cleanly divided into **3 dedicated files** inside the `modules/` folder:

```text
Student_Performance_Streamlit/
│
├── data/
│   ├── student-mat.csv               # Mathematics course dataset (395 students)
│   └── student-por.csv               # Portuguese language dataset (649 students)
│
├── Models/
│   ├── app_features.pkl              # Feature vector definition
│   └── student_prediction_app_model.pkl # Trained Random Forest Classifier
│
├── modules/
│   ├── member1_data_analysis.py      # [MEMBER 1] Data loading, preprocessing & EDA charts
│   ├── member2_ml_models.py          # [MEMBER 2] ML prediction, model evaluation & benchmarks
│   └── member3_ui_views.py           # [MEMBER 3] Streamlit UI, styling & What-If simulator
│
├── app.py                            # Main Streamlit app uniting all 3 modules
├── Student_Performance_Prediction.ipynb # Jupyter Notebook for model training & experiments
├── requirements.txt                  # Python package dependencies
├── README.md                         # Documentation
└── .gitignore                        # Git ignore rules
```

---

## Team Roles & Ownership

### Member 1: Dataset Analysis, Preprocessing & Visualizations
* **File:** `modules/member1_data_analysis.py`
* **Responsibilities:**
  * Loads and cleans the UCI dataset (`data/student-mat.csv`).
  * Computes cohort statistics (pass rate, average marks, absences).
  * Generates interactive charts: Pass vs. Fail ratio, exam score distributions ($G_1, G_2, G_3$), study hours impact, and correlation heatmap.
* **Dashboard Tab:** **Data Analysis & Graphs**

### Member 2: Machine Learning Models, Evaluation & Prediction
* **File:** `modules/member2_ml_models.py`
* **Responsibilities:**
  * Model inference using the Random Forest classifier (`Models/student_prediction_app_model.pkl`).
  * Computes quantitative evaluation metrics (Accuracy: **94.18%**, Precision: **97.29%**, Recall: **94.72%**, F1: **95.98%**).
  * Real Confusion Matrix: 121 TN, 251 TP, 14 FP, 9 FN.
  * 5-Fold Cross-Validation benchmark against Decision Tree (86.84%) and Logistic Regression (90.38%).
  * Feature importance analysis ($G_2$: 58.3%, $G_1$: 28.8%, past failures: 5.9%, absences: 5.3%, study time: 1.7%).
* **Dashboard Tab:** **Model Evaluation & Results**

### Member 3: Dashboard Design, Frontend & Integration
* **File:** `modules/member3_ui_views.py` and `app.py`
* **Responsibilities:**
  * Main application flow and custom CSS styling.
  * Sidebar navigation with preset student personas (*Top Student*, *Average Student*, *Struggling Student*, *Comeback Student*, *Attendance Risk Student*).
  * Tab 1: Live student assessment form, KPI metric cards, and 1-click report generator (.txt).
  * Tab 4: Interactive What-If Habit Sensitivity Simulator.
* **Dashboard Tabs:** **Student Pass Predictor** & **What-If Habit Simulator**

---

## Running the Application

### 1. Clone the Repository
```bash
git clone https://github.com/AkanshaTomar830/Student_Performance_Streamlit.git
cd Student_Performance_Streamlit
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run with Streamlit
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## License & Dataset Source
Trained on the **UCI Student Performance Dataset** (Cortez & Silva, 2008) for academic Project-Based Learning (PBL) demonstration.
