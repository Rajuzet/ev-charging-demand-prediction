# ⚡ EV Charging Demand Prediction and Usage Pattern Analysis

An end-to-end Machine Learning, Time Series, Explainable AI (XAI), and Streamlit application for predicting EV charging station demand, identifying high-demand sessions, handling class imbalances, and forecasting future energy requirements.

---

## 📌 Executive Summary & Business Use Cases

Electric Vehicle (EV) adoption requires smart charging infrastructure management to avoid grid overloads, reduce driver waiting times, and optimize operational expenditure. This project delivers an industrial-grade ML solution covering:

1. **Charging Station Demand Forecasting:** Predicting session energy demand (kWh) based on duration, location, charger type, operator, and temporal features.
2. **High-Demand Session Classification:** Classifying whether incoming charging sessions belong to **High Demand** ($\ge 14.2$ kWh) vs. **Normal Demand** to support load balancing.
3. **Class Imbalance Mitigation:** Evaluating **Random Oversampling, SMOTE, and ADASYN** to ensure fair minority-class recall.
4. **Time Series Forecasting:** Analyzing multi-month historical trends, testing stationarity (**Augmented Dickey-Fuller Test**), and forecasting with **Moving Average & Exponential Smoothing (Holt-Winters)**.
5. **Explainable AI (XAI):** Leveraging **SHAP (TreeExplainer)** and **LIME (Local Interpretable Model-agnostic Explanations)** to explain why specific predictions were made.
6. **Interactive Streamlit Dashboard:** Deploying a multi-tab web application for real-time predictions, analytics, and operational decision-making.

---

## 🏗️ Project Architecture & Directory Structure

```text
ev_charging_demand_analysis/
│
├── data/
│   ├── raw/                           # Original raw dataset (ChargingRecords.csv)
│   └── processed/                     # Cleaned & feature-engineered dataset
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb # Exploratory Data Analysis & visual insights
│   └── 02_Machine_Learning_Demand_Prediction.ipynb # ML benchmarks & XAI
│
├── src/
│   ├── preprocessing.py               # Data cleaning & feature engineering pipeline
│   ├── train.py                       # ML models training, evaluation & serialization
│   └── create_notebooks.py            # Reproducible notebook generator
│
├── models/                            # Serialized models & artifacts (.pkl)
│   ├── regression_model.pkl           # Best Regression model
│   ├── classification_model.pkl       # Best Classification model
│   ├── preprocessor.pkl               # OneHotEncoder & StandardScaler pipeline
│   ├── timeseries_model.pkl           # Holt-Winters Forecasting model
│   ├── evaluation_metrics.json        # Benchmark metrics across all models
│   ├── daily_timeseries.csv           # Resampled daily demand
│   └── shap_sample_data.csv           # Representative sample for XAI explanations
│
├── app/
│   └── app.py                         # Interactive Streamlit Web Application
│
├── requirements.txt                   # Complete Python dependencies
├── .gitignore                         # Git exclusion rules
└── README.md                          # Project documentation
```

---

## 📊 Dataset Overview

- **Source:** Public EV charging transaction records
- **Total Sessions:** 72,856 raw records (~72,421 after anomaly cleaning)
- **Key Features:**
  - `Location`: Station venue (hotel, company, apartment, resort, public area, etc.)
  - `ChargerCompany`: Infrastructure operator ID (0, 1)
  - `ChargerType`: Charger specification (0: Standard AC, 1: Fast DC)
  - `StartDatetime` / `EndDatetime`: Timestamps of charging session
  - `Duration_Mins`: Verified duration in minutes
  - `Demand`: Continuous energy consumed in kWh (**Regression Target**)
  - `High_Demand`: Binary indicator (1 if Demand $\ge 14.20$ kWh median, else 0) (**Classification Target**)

---

## 🚀 Setup & Installation

### 1. Clone or Open Project
```bash
cd ev_charging_demand_analysis
```

### 2. Create and Activate Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ How to Run

### Step 1: Run Data Preprocessing Pipeline
Cleans raw data, recalculates true durations, extracts temporal features, and generates `cleaned_charging_records.csv`:
```bash
python src/preprocessing.py
```

### Step 2: Train and Benchmark Machine Learning Models
Trains all 6 regression models, 7 classification models, imbalance resampling techniques, and time series forecasting models:
```bash
python src/train.py
```

### Step 3: Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```

### Step 4: Run Jupyter Notebooks
```bash
jupyter notebook notebooks/01_EDA_and_Preprocessing.ipynb
```

---

## 📈 Machine Learning Benchmark Results

### 1. Regression Models (Predicting Demand in kWh)
| Model | MAE (kWh) | RMSE (kWh) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Random Forest** | **3.37** | **6.08** | **0.7983** |
| **XGBoost** | 3.43 | 6.12 | 0.7956 |
| **LightGBM** | 3.47 | 6.16 | 0.7929 |
| **Gradient Boosting** | 3.51 | 6.24 | 0.7878 |
| **Decision Tree** | 3.52 | 6.75 | 0.7518 |
| **Linear Regression** | 5.61 | 8.69 | 0.5882 |

### 2. Classification Models (High-Demand vs Normal-Demand)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.875** | **0.869** | **0.884** | **0.876** | **0.949** |
| **XGBoost** | 0.874 | 0.867 | 0.884 | 0.875 | 0.948 |
| **Random Forest** | 0.871 | 0.862 | 0.883 | 0.872 | 0.946 |
| **Decision Tree** | 0.852 | 0.849 | 0.856 | 0.852 | 0.893 |
| **SVM (LinearSVC)** | 0.828 | 0.817 | 0.846 | 0.831 | 0.912 |
| **Logistic Regression** | 0.827 | 0.815 | 0.846 | 0.830 | 0.911 |
| **KNN** | 0.819 | 0.806 | 0.840 | 0.823 | 0.895 |

### 3. Class Imbalance Handling Comparison
Evaluated on peak demand minority sessions:
- **Baseline (No Resampling):** F1: 0.682 | Recall: 0.621 | ROC-AUC: 0.864
- **Random Oversampling:** F1: 0.714 | Recall: 0.782 | ROC-AUC: 0.871
- **SMOTE:** F1: 0.718 | Recall: 0.795 | ROC-AUC: 0.873
- **ADASYN:** F1: 0.716 | Recall: 0.804 | ROC-AUC: 0.872

### 4. Time Series Forecasting
- **Stationarity:** Augmented Dickey-Fuller Test ($p < 0.01$, confirms series is stationary).
- **Models Evaluated:**
  - 7-Day Moving Average (MAE: ~1,140 kWh, RMSE: ~1,450 kWh)
  - Simple Exponential Smoothing (MAE: ~1,120 kWh, RMSE: ~1,430 kWh)
  - **Holt-Winters Exponential Smoothing** (Best multi-step forecasting with weekly seasonality)

---

## 🔍 Explainable AI (SHAP & LIME)

- **TreeExplainer (SHAP):** Identifies `Duration_Mins`, `ChargerType_Avg_Demand`, `StartHour`, and `Loc_Avg_Demand` as the top 4 drivers of charging load globally.
- **LIME:** Provides tabular local explanations showing exact feature condition contributions (+kWh or -kWh) for individual customer sessions.

---

## 🌐 Streamlit Dashboard Features

1. **Executive KPIs & EDA:** Real-time metrics, location breakdowns, day-of-week trends, and peak hour distributions.
2. **Regression Predictor:** Interactive session simulator that outputs predicted energy consumption (kWh) with comparative gauge.
3. **High-Demand Classifier:** Predicts high-load sessions and outputs actionable grid balancing suggestions.
4. **Time Series Forecasting:** Visualizes historical demand with 7-day/30-day moving averages and interactive Holt-Winters forecasts.
5. **Explainable AI (XAI):** Live LIME local explanation and SHAP feature importance charts.
6. **Model Evaluation Benchmark:** Complete leaderboard comparing all models and imbalance strategies.

---

## 📦 Pushing to Your GitHub Repository

```bash
# 1. Check status
git status

# 2. Stage all project files
git add .

# 3. Commit changes
git commit -m "feat: complete end-to-end EV charging demand prediction and analytics solution"

# 4. Link to your personal GitHub repository
git remote add origin https://github.com/<your-username>/ev-charging-demand-prediction.git
git branch -M main
git push -u origin main
```
