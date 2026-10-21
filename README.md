# StreamGuard: OTT Customer Churn & Revenue Retention Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://palletsprojects.com/p/flask/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

> **StreamGuard** is an end-to-end Machine Learning intelligence platform designed for Over-The-Top (OTT) streaming platforms (e.g., Netflix, Prime Video, Disney+). It features dual AI missions: **Subscriber Churn Classification** (predicting cancellation with optimal threshold tuning) and **Customer Lifetime Value (CLV) Continuous Regression** (forecasting monetary value and prioritizing revenue at risk).

---

## 🌟 Key Capabilities & System Highlights

1. **Dual Machine Learning Capabilities**:
   - **Primary Classification Model:** Tuned Logistic Regression pipeline delivering **0.8452 Held-Out Test ROC-AUC** and **66.7% Recall** at an optimized **0.60 decision threshold**.
   - **Secondary Continuous Regression Model:** Random Forest Regressor achieving **$R^2 = 0.7672$** and **$89.16 MAE** for Customer Lifetime Value ($USD) forecasting.
2. **Leakage-Safe Data Engineering**:
   - Automated IQR outlier clipping (`IQRClipper`) learned strictly on the training partition.
   - Domain-specific engineered features: **Engagement Index** (`Watch_Hours * (Login_Frequency / 30)`) and **Watch-to-Fee Ratio** (`Watch_Hours / Monthly_Fee`).
   - Stratified 80/20 train/test partition with 5-fold cross-validation.
3. **Executive Web Intelligence Dashboard**:
   - **Live Telemetry & KPIs:** Active subscribers (1,503), Platform Churn Rate (17.03%), Total Revenue-at-Risk ($104,152), Average CLV ($406.26).
   - **Interactive AI Predictor:** Real-time single subscriber risk diagnosis with interactive sliders, 1-click test scenarios, animated radial probability gauge, and prescriptive AI retention protocols.
   - **Subscriber Cohort Explorer:** Searchable, filterable directory with pagination across all 1,503 cleaned subscriber profiles with instant "1-Click Load into Predictor".
   - **Batch CSV Inference Engine:** Drag-and-drop CSV upload for portfolio risk audits, breakdown charts, and downloadable enriched CSVs.
   - **Pipeline Intelligence Center:** Interactive 5-Model cross-validation benchmark table, confusion matrix inspection (206 TN, 44 FP, 17 FN, 34 TP), and real-time decision threshold trade-off simulator.

---

## 📁 Repository Directory Structure

```text
Streamguard/
├── app.py                      # Production Flask REST API and static server
├── run.py                      # Standalone launcher script (http://127.0.0.1:5000)
├── streamguard_transformers.py  # Modular scikit-learn transformers (OTTFeatureEngineer, IQRClipper)
├── requirements.txt            # Python dependencies
├── README.md                   # Complete architectural documentation
├── data/
│   ├── ott_customer_churn_raw.csv      # Original raw OTT subscriber dataset (1,505 rows)
│   └── ott_customer_churn_cleaned.csv  # Sanitized, verified dataset (1,503 rows)
├── models/
│   ├── churn_classifier.joblib # Serialized winning churn classification pipeline
│   ├── clv_regressor.joblib    # Serialized Random Forest CLV regressor
│   └── pipeline_metadata.json  # Benchmarks, test metrics, and feature metadata
├── notebook/
│   └── model.ipynb             # 73-cell end-to-end Jupyter Notebook with verified outputs
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphism dark-mode master design system
│   └── js/
│       └── app.js              # Modular client-side interactive telemetry & Chart.js logic
└── templates/
    └── index.html              # Modern semantic SPA layout
```

---

## 🔬 Machine Learning Pipeline (5 Modules)

The Jupyter notebook ([notebook/model.ipynb](notebook/model.ipynb)) is structured into 5 cohesive modules:

| Module | Core Mission | Key Implementation Steps | Test Results |
|---|---|---|---|
| **Module 1** | Data Wrangling & Outlier Auditing | Resilient multi-path CSV loading, IQR outlier bounds, negative watch/fee sanitization, cleaned dataset export | 1,503 clean records saved |
| **Module 2** | Feature Engineering & Pipelines | Custom `OTTFeatureEngineer` (Engagement Metric, Watch-to-Fee Ratio), `IQRClipper`, `StandardScaler`, `OneHotEncoder` | 24 processed leak-free features |
| **Module 3** | Validation & Feature Selection | Stratified 80/20 train/test split, 5-Fold Stratified K-Fold splitter, RFE benchmark | Full 24 feature set confirmed superior |
| **Module 4** | Baselines & Hyperparameter Tuning | Logistic Regression, KNN, SVM baselines; RandomizedSearchCV for Random Forest and AdaBoost; CLV regression benchmark | Baseline CV comparison established |
| **Module 5** | Nested CV, Thresholding & Deployment | 5-Model nested CV, decision threshold optimization (0.60), confusion matrix (206 TN, 44 FP, 17 FN, 34 TP), production artifact serialization | **AUC 0.8452, Recall 66.7%, R² 0.7672** |

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Platform
```bash
python run.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Run the Jupyter Notebook
Open [notebook/model.ipynb](notebook/model.ipynb) in VS Code or JupyterLab:
```bash
jupyter lab notebook/model.ipynb
```
*Note: All 73 cells have been executed and validated with 100% clean outputs.*

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/status` | GET | System status, model health, and headline KPI metrics |
| `GET /api/overview-stats` | GET | Aggregates by plan tier, device, payment method, and tenure cohorts |
| `GET /api/model-info` | GET | 5-Model benchmark metrics, confusion matrix, and threshold trade-off curve points |
| `GET /api/subscribers` | GET | Paginated, searchable, and filterable list of subscribers |
| `POST /api/predict` | POST | Real-time single subscriber churn probability, predicted CLV, and prescriptive retention actions |
| `POST /api/batch-predict` | POST | Multi-subscriber CSV file upload inference and risk portfolio calculation |
| `GET /api/download-sample-csv` | GET | Download sample multi-subscriber CSV for testing batch predictions |
| `GET /api/export-cleaned` | GET | Download the verified 1,503-record cleaned dataset |
