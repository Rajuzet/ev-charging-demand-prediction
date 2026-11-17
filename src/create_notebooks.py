"""
Script to generate well-documented, reproducible Jupyter Notebooks for the Capstone project.
"""

import json
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notebooks_dir = os.path.join(base_dir, 'notebooks')
os.makedirs(notebooks_dir, exist_ok=True)

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {filepath}")

# 1. EDA and Preprocessing Notebook
cells_eda = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# EV Charging Demand Prediction & Usage Pattern Analysis\n",
            "## Part 1: Data Understanding, Cleaning & Exploratory Data Analysis (EDA)\n",
            "\n",
            "### Objectives:\n",
            "- Inspect and understand the EV charging dataset (~72,000+ sessions)\n",
            "- Clean anomalies, handle duplicates, correct datetime formats, and validate durations\n",
            "- Analyze charging demand patterns across locations, charger types, and companies\n",
            "- Uncover peak charging hours, day-of-week trends, and feature correlations"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import required libraries\n",
            "import os\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
            "plt.rcParams['figure.figsize'] = (12, 6)\n",
            "print(\"Libraries loaded successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Load Raw Dataset"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "raw_path = os.path.join('..', 'data', 'raw', 'ChargingRecords.csv')\n",
            "df_raw = pd.read_csv(raw_path)\n",
            "print(f\"Raw dataset shape: {df_raw.shape}\")\n",
            "df_raw.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Dataset Information & Missing Value Assessment"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Data Types & Non-Null Counts:\")\n",
            "df_raw.info()\n",
            "print(\"\\nMissing values per column:\\n\", df_raw.isnull().sum())\n",
            "print(f\"\\nDuplicate rows count: {df_raw.duplicated().sum()}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Data Cleaning Pipeline\n",
            "- Deduplication\n",
            "- Datetime conversion\n",
            "- Duration recalculation & anomaly filtering"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Deduplicate\n",
            "df_clean = df_raw.drop_duplicates().copy()\n",
            "\n",
            "# Datetime parsing\n",
            "df_clean['StartDatetime'] = pd.to_datetime(df_clean['StartDatetime'], errors='coerce')\n",
            "df_clean['EndDatetime'] = pd.to_datetime(df_clean['EndDatetime'], errors='coerce')\n",
            "\n",
            "# Recalculate duration in minutes\n",
            "calc_duration = (df_clean['EndDatetime'] - df_clean['StartDatetime']).dt.total_seconds() / 60.0\n",
            "df_clean['Duration_Mins'] = np.where(df_clean['Duration'] > 0, df_clean['Duration'], calc_duration)\n",
            "\n",
            "# Filter realistic duration and demand records\n",
            "valid_mask = (df_clean['Duration_Mins'] > 0) & (df_clean['Duration_Mins'] <= 1440) & (df_clean['Demand'] > 0)\n",
            "df_clean = df_clean[valid_mask].reset_index(drop=True)\n",
            "\n",
            "print(f\"Cleaned records: {len(df_clean):,}\")\n",
            "df_clean[['Duration_Mins', 'Demand']].describe()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. Feature Engineering\n",
            "Extract hour, day, month, day of week, cyclical sine/cosine features, and classification median threshold."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_clean['StartHour'] = df_clean['StartDatetime'].dt.hour\n",
            "df_clean['StartDay'] = df_clean['StartDatetime'].dt.day\n",
            "df_clean['StartMonth'] = df_clean['StartDatetime'].dt.month\n",
            "df_clean['DayOfWeek'] = df_clean['StartDatetime'].dt.day_of_week\n",
            "df_clean['DayName'] = df_clean['StartDatetime'].dt.day_name()\n",
            "df_clean['IsWeekend'] = df_clean['DayOfWeek'].isin([5, 6]).astype(int)\n",
            "\n",
            "# Cyclical hour encoding\n",
            "df_clean['Hour_Sin'] = np.sin(2 * np.pi * df_clean['StartHour'] / 24.0)\n",
            "df_clean['Hour_Cos'] = np.cos(2 * np.pi * df_clean['StartHour'] / 24.0)\n",
            "\n",
            "# Classification target per project specifications: High vs Normal Demand\n",
            "median_demand = float(df_clean['Demand'].median())\n",
            "df_clean['High_Demand'] = (df_clean['Demand'] >= median_demand).astype(int)\n",
            "\n",
            "print(f\"Median Demand Threshold: {median_demand:.2f} kWh\")\n",
            "print(df_clean['High_Demand'].value_counts(normalize=True))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5. Exploratory Visualizations"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Demand by Location\n",
            "plt.figure(figsize=(14, 5))\n",
            "sns.barplot(data=df_clean, x='Location', y='Demand', estimator=np.sum, errorbar=None, palette='Blues_r')\n",
            "plt.title('Total EV Charging Energy Delivered by Location (kWh)', fontsize=14, fontweight='bold')\n",
            "plt.xticks(rotation=45)\n",
            "plt.ylabel('Total Demand (kWh)')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Hourly Charging Demand Pattern\n",
            "hourly_agg = df_clean.groupby('StartHour')['Demand'].agg(['count', 'mean']).reset_index()\n",
            "\n",
            "fig, ax1 = plt.subplots(figsize=(14, 5))\n",
            "ax2 = ax1.twinx()\n",
            "\n",
            "sns.barplot(data=hourly_agg, x='StartHour', y='count', ax=ax1, color='#93C5FD', alpha=0.7)\n",
            "sns.lineplot(data=hourly_agg, x='StartHour', y='mean', ax=ax2, color='#1D4ED8', marker='o', linewidth=2.5)\n",
            "\n",
            "ax1.set_title('EV Charging Session Volume & Average Demand Across Hours', fontsize=14, fontweight='bold')\n",
            "ax1.set_xlabel('Hour of Day (0-23)')\n",
            "ax1.set_ylabel('Session Volume', color='#1E3A8A')\n",
            "ax2.set_ylabel('Mean Demand (kWh)', color='#1D4ED8')\n",
            "plt.grid(False)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Charger Type Comparison\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "sns.countplot(data=df_clean, x='ChargerType', ax=axes[0], palette='Set2')\n",
            "axes[0].set_title('Session Count by Charger Type (0 = AC, 1 = DC Fast)')\n",
            "\n",
            "sns.boxplot(data=df_clean, x='ChargerType', y='Demand', ax=axes[1], palette='Set2', showfliers=False)\n",
            "axes[1].set_title('Demand Distribution by Charger Type')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

create_notebook(cells_eda, os.path.join(notebooks_dir, '01_EDA_and_Preprocessing.ipynb'))

# 2. Machine Learning End-to-End Notebook
cells_ml = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# EV Charging Demand Prediction & Usage Pattern Analysis\n",
            "## Part 2: Regression, Classification, Time Series Forecasting & XAI\n",
            "\n",
            "### Contents:\n",
            "1. **Regression:** Predict continuous Demand (kWh) using Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM (evaluated with MAE, MSE, RMSE, R²).\n",
            "2. **Classification:** Predict High-Demand vs Normal-Demand sessions, evaluate class imbalance techniques (Random Oversampling, SMOTE, ADASYN), and benchmark 7 algorithms.\n",
            "3. **Time Series Forecasting:** Check ADF stationarity, Moving Average, and Exponential Smoothing (evaluated with MAE & RMSE).\n",
            "4. **Explainable AI:** Feature Importance, SHAP, and LIME interpretability."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import json\n",
            "import joblib\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix\n",
            "\n",
            "print(\"Libraries ready!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Load Processed Dataset & Models"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_path = os.path.join('..', 'data', 'processed', 'cleaned_charging_records.csv')\n",
            "df = pd.read_csv(data_path)\n",
            "print(f\"Loaded clean data: {df.shape}\")\n",
            "\n",
            "# Load saved metrics summary\n",
            "with open(os.path.join('..', 'models', 'evaluation_metrics.json'), 'r') as f:\n",
            "    metrics = json.load(f)\n",
            "\n",
            "print(f\"Best Regression Model: {metrics['best_regression_model']}\")\n",
            "print(f\"Best Classification Model: {metrics['best_classification_model']}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Regression Models Performance Comparison"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "reg_df = pd.DataFrame(metrics['regression_metrics']).T.sort_values(by='R2', ascending=False)\n",
            "print(\"=== Regression Model Comparison (Target: Demand kWh) ===\")\n",
            "reg_df"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 5))\n",
            "sns.barplot(x=reg_df.index, y=reg_df['R2'], palette='crest')\n",
            "plt.title('Regression Models R² Score Comparison', fontsize=14, fontweight='bold')\n",
            "plt.ylabel('R² Score')\n",
            "plt.xticks(rotation=30)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Classification Models Benchmark & Imbalance Handling"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "clf_df = pd.DataFrame(metrics['classification_metrics']).T.sort_values(by='F1', ascending=False)\n",
            "print(\"=== Classification Model Comparison (Target: High Demand) ===\")\n",
            "clf_df"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "imb_df = pd.DataFrame(metrics['imbalance_comparison']).T\n",
            "print(\"=== Class Imbalance Technique Comparison ===\")\n",
            "imb_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. Time Series Forecasting & Trend Analysis\n",
            "- Augmented Dickey-Fuller (ADF) Stationarity Test\n",
            "- Moving Average (7-Day & 30-Day Rolling)\n",
            "- Additive Seasonal Decomposition (Trend, Seasonality, Noise)\n",
            "- Exponential Smoothing (Simple & Holt-Winters Seasonal)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "daily_ts = pd.read_csv(os.path.join('..', 'models', 'daily_timeseries.csv'))\n",
            "daily_ts['Date'] = pd.to_datetime(daily_ts['Date'])\n",
            "daily_ts['7D_MA'] = daily_ts['Total_Demand_kWh'].rolling(7).mean()\n",
            "daily_ts['30D_MA'] = daily_ts['Total_Demand_kWh'].rolling(30).mean()\n",
            "\n",
            "plt.figure(figsize=(14, 5))\n",
            "plt.plot(daily_ts['Date'], daily_ts['Total_Demand_kWh'], color='lightgray', alpha=0.7, label='Actual Daily Demand (kWh)')\n",
            "plt.plot(daily_ts['Date'], daily_ts['7D_MA'], color='#0284c7', linewidth=2.5, label='7-Day Moving Average')\n",
            "plt.plot(daily_ts['Date'], daily_ts['30D_MA'], color='#e11d48', linewidth=2, label='30-Day Moving Average')\n",
            "plt.title('Daily EV Charging Demand Trend & Moving Averages (366 Days)', fontsize=14, fontweight='bold')\n",
            "plt.xlabel('Date')\n",
            "plt.ylabel('Total Daily Energy (kWh)')\n",
            "plt.legend()\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "decomp_path = os.path.join('..', 'models', 'timeseries_decomposition.csv')\n",
            "if os.path.exists(decomp_path):\n",
            "    decomp_df = pd.read_csv(decomp_path)\n",
            "    decomp_df['Date'] = pd.to_datetime(decomp_df['Date'])\n",
            "    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)\n",
            "    axes[0].plot(decomp_df['Date'], decomp_df['Trend'], color='#0284c7', linewidth=2)\n",
            "    axes[0].set_ylabel('Trend (kWh)')\n",
            "    axes[0].set_title('Time Series Decomposition: Trend, 7-Day Seasonality & Noise', fontsize=14, fontweight='bold')\n",
            "    axes[1].plot(decomp_df['Date'], decomp_df['Seasonal'], color='#10b981', linewidth=1.5)\n",
            "    axes[1].set_ylabel('Seasonal (kWh)')\n",
            "    axes[2].plot(decomp_df['Date'], decomp_df['Residual'], color='#64748b', linewidth=1)\n",
            "    axes[2].set_ylabel('Residual (kWh)')\n",
            "    axes[2].set_xlabel('Date')\n",
            "    plt.tight_layout()\n",
            "    plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "ts_models = pd.DataFrame(metrics['timeseries_metrics']['Models']).T\n",
            "print('=== Time Series Forecasting Benchmark ===')\n",
            "print(f\"ADF Test Statistic : {metrics['timeseries_metrics']['Stationarity_ADF']['ADF_Statistic']:.4f}\")\n",
            "print(f\"ADF p-value        : {metrics['timeseries_metrics']['Stationarity_ADF']['p_value']:.4e}\")\n",
            "print(f\"Is Stationary      : {metrics['timeseries_metrics']['Stationarity_ADF']['Is_Stationary']} (Non-Stationary Growth Trend)\")\n",
            "ts_models"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5. Explainable AI (SHAP & LIME)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import shap\n",
            "import lime\n",
            "from lime.lime_tabular import LimeTabularExplainer\n",
            "\n",
            "reg_model = joblib.load(os.path.join('..', 'models', 'regression_model.pkl'))\n",
            "preprocessor = joblib.load(os.path.join('..', 'models', 'preprocessor.pkl'))\n",
            "shap_sample = pd.read_csv(os.path.join('..', 'models', 'shap_sample_data.csv'))\n",
            "\n",
            "feature_cols = metrics['categorical_columns'] + metrics['numerical_columns']\n",
            "X_sample_proc = preprocessor.transform(shap_sample[feature_cols])\n",
            "\n",
            "cat_encoder = preprocessor.named_transformers_['cat']\n",
            "cat_features = cat_encoder.get_feature_names_out(metrics['categorical_columns'])\n",
            "all_features = list(cat_features) + metrics['numerical_columns']\n",
            "\n",
            "print(\"Top 10 Feature Importances:\")\n",
            "feat_imp = pd.Series(reg_model.feature_importances_, index=all_features).sort_values(ascending=False).head(10)\n",
            "print(feat_imp)"
        ]
    }
]

create_notebook(cells_ml, os.path.join(notebooks_dir, '02_Machine_Learning_Demand_Prediction.ipynb'))
print("Notebooks generation script finished.")
