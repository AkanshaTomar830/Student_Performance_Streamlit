# EduPredict AI — Student Performance Prediction & Analytics Dashboard

An end-to-end, modular Machine Learning decision-support system built with **Streamlit** and a **Random Forest Classifier** trained on the UCI Student Performance Dataset.

The platform helps educators, academic advisors, and students assess course performance, identify at-risk students before final examinations, explore data trends, compare machine learning algorithms, and simulate habit improvements.

---

## Team Work Breakdown (3-Member Modular Architecture)

This project has been modularized into 3 independent components, enabling 3 team members to develop and defend their contributions separately:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   EduPredict AI                                        │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│        MEMBER 1          │          MEMBER 2           │           MEMBER 3            │
│  Data Analytics & EDA    │    ML Pipeline & Models     │      Dashboard & UI/UX        │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ • Dataset ingestion      │ • Production predictor      │ • Main Streamlit orchestrator │
│ • Data preprocessing     │ • Model performance metrics │ • Custom CSS styling & themes │
│ • Feature correlation    │ • Confusion matrix analysis │ • Sidebar controls & personas │
│ • Interactive charts     │ • 5-fold cross-validation   │ • Multi-tab view integration  │
│ • EDA cohort insights    │ • Feature importance (XAI)  │ • Report export & downloads   │
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

### Member 1: Dataset Analysis, Preprocessing & Visualizations
* **Directory:** `src/data_analytics/`
* **Modules:**
  * `loader.py`: Ingestion of benchmark dataset (`data/student-mat.csv`), missing value validation, and cohort metrics.
  * `preprocessing.py`: Feature selection, binary classification labeling ($G_3 \ge 10$), and correlation calculations.
  * `visualizations.py`: Exploratory Data Analysis (EDA) charts including cohort pass/fail distribution, grade histograms across $G_1, G_2, G_3$, study hours impact, and correlation heatmaps.
* **Dashboard Tab:** **Dataset Analytics & EDA**

### Member 2: Machine Learning Models, Evaluation & Prediction
* **Directory:** `src/ml_pipeline/`
* **Modules:**
  * `predictor.py`: Production inference engine using the trained Random Forest ensemble (`Models/student_prediction_app_model.pkl`), probability confidence scoring, and risk classification.
  * `evaluator.py`: Quantitative model evaluation (Confusion Matrix: 121 TN, 251 TP, 14 FP, 9 FN; Accuracy: 94.18%, Precision: 97.29%, F1 Score: 95.98%) and 5-fold cross-validation benchmarking against Decision Trees and Logistic Regression.
  * `explainability.py`: Explainable AI (XAI) feature importances ($G_2$: 58.3%, $G_1$: 28.8%, past failures: 5.9%, absences: 5.3%, study time: 1.7%).
* **Dashboard Tab:** **Model Evaluation & XAI**

### Member 3: Dashboard Design, Frontend & System Integration
* **Directory:** `src/ui_components/` and `app.py`
* **Modules:**
  * `styles.py`: Clean CSS themes, responsive hero banners, and KPI metric card components.
  * `sidebar.py`: Quick-load student personas (*Top Student*, *Average Student*, *Struggling Student*, *Comeback Student*, *Attendance Risk Student*), student identifiers, and team credits.
  * `views/view_prediction.py`: Interactive assessment form, visual outcome probability bar, and downloadable evaluation report (.txt).
  * `views/view_simulation.py`: What-If Sensitivity Simulator modeling real-time gains from increasing study hours or decreasing absences.
  * `app.py`: Main application orchestrator uniting Member 1, Member 2, and Member 3 into a single responsive portal.
* **Dashboard Tabs:** **Student Pass Predictor** & **What-If Habit Simulator**

---

## Project Structure

```text
Student_Performance_Streamlit/
│
├── data/                                 # [Member 1] Dataset Storage
│   ├── student-mat.csv                   # Mathematics course records (395 students)
│   └── student-por.csv                   # Portuguese language records (649 students)
│
├── src/                                  # Modular Source Code
│   ├── __init__.py
│   │
│   ├── data_analytics/                   # [Member 1 Ownership]
│   │   ├── __init__.py
│   │   ├── loader.py                     # Data loading & summary statistics
│   │   ├── preprocessing.py              # Feature extraction & correlation matrix
│   │   └── visualizations.py             # Altair EDA charts & distribution plots
│   │
│   ├── ml_pipeline/                      # [Member 2 Ownership]
│   │   ├── __init__.py
│   │   ├── predictor.py                  # Production inference & scenario engine
│   │   ├── evaluator.py                  # Confusion matrix & model benchmarking
│   │   └── explainability.py             # Feature importance analysis (XAI)
│   │
│   └── ui_components/                    # [Member 3 Ownership]
│       ├── __init__.py
│       ├── styles.py                     # CSS styling & responsive card themes
│       ├── sidebar.py                    # Sidebar controls & persona presets
│       └── views/                        # Feature view components
│           ├── __init__.py
│           ├── view_prediction.py        # Tab 1: Live Student Predictor
│           ├── view_analytics.py         # Tab 2: Dataset Analytics & EDA
│           ├── view_models.py            # Tab 3: Model Evaluation & XAI
│           └── view_simulation.py        # Tab 4: What-If Habit Simulator
│
├── Models/                               # Production Model Artifacts
│   ├── app_features.pkl                  # Feature vector definition
│   └── student_prediction_app_model.pkl  # Trained Random Forest Classifier
│
├── .vscode/
│   └── launch.json                      # One-click VS Code Run & Debug configuration
├── app.py                               # Main Streamlit Orchestrator
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation
```

---

## Machine Learning Benchmark Results

Evaluated on the UCI Student Performance benchmark dataset:

| Model | 5-Fold CV Accuracy | 5-Fold CV F1 Score | Test Set Accuracy |
| :--- | :---: | :---: | :---: |
| **Random Forest (Champion)** | **90.38%** | **93.18%** | **94.18%** |
| **Logistic Regression** | 90.38% | 93.07% | 91.90% |
| **Decision Tree** | 86.84% | 90.41% | 91.39% |

### Confusion Matrix (Random Forest):
* **True Negatives (Correctly predicted Fail):** 121
* **True Positives (Correctly predicted Pass):** 251
* **False Positives (Incorrectly predicted Pass):** 14
* **False Negatives (Incorrectly predicted Fail):** 9

---

## Installation & Running

### 1. Clone the Repository
```bash
git clone https://github.com/AkanshaTomar830/Student_Performance_Streamlit.git
cd Student_Performance_Streamlit
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
python -m streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## License & Credits
Developed as an academic Project-Based Learning (PBL) collaboration utilizing the **UCI Student Performance Dataset** (Cortez & Silva, 2008).
