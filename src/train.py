"""
Model Training and Evaluation Pipeline
EV Charging Demand Prediction and Usage Pattern Analysis
Covers:
1. Regression (Demand in kWh)
2. Classification (High vs Normal Demand) + Imbalance handling (SMOTE, ADASYN, ROS)
3. Time Series Forecasting (Moving Average & Exponential Smoothing)
4. Model Explainability Preparation (SHAP & LIME)
5. Model Persistence (.pkl)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Imbalance Techniques
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# Time Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing


def load_processed_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])
    df['EndDatetime'] = pd.to_datetime(df['EndDatetime'])
    return df


def prepare_features(df: pd.DataFrame):
    """
    Select relevant features avoiding target leakage.
    Demand and Avg_Power_kW are excluded from feature set.
    """
    categorical_cols = ['Location', 'ChargerCompany', 'ChargerType', 'TimeOfDay', 'IsWeekend']
    numerical_cols = ['Duration_Mins', 'StartHour', 'StartDay', 'StartMonth', 'DayOfWeek', 
                      'Hour_Sin', 'Hour_Cos', 'Loc_Avg_Demand', 'ChargerType_Avg_Demand']

    feature_cols = categorical_cols + numerical_cols
    X = df[feature_cols].copy()
    y_reg = df['Demand'].copy()
    y_clf = df['High_Demand'].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )

    return X, y_reg, y_clf, preprocessor, categorical_cols, numerical_cols


def train_regression_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate 6 regression models per requirements."""
    print("\n--- Training Regression Models (Demand Prediction) ---")
    
    # Fit preprocessor
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=12, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    }

    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model = None

    for name, model in models.items():
        print(f"  Training {name}...")
        if name == 'KNN':
            # Train KNN on stratified sample of 15,000 for responsive nearest-neighbor inference
            knn_idx = np.random.RandomState(42).choice(len(X_train_proc), size=min(15000, len(X_train_proc)), replace=False)
            model.fit(X_train_proc[knn_idx], y_train.values[knn_idx])
        else:
            model.fit(X_train_proc, y_train)
        preds = model.predict(X_test_proc)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        results[name] = {
            'MAE': round(float(mae), 4),
            'MSE': round(float(mse), 4),
            'RMSE': round(float(rmse), 4),
            'R2': round(float(r2), 4)
        }
        print(f"    {name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model

    print(f"Best Regression Model: {best_model_name} (R2 = {best_r2:.4f})")
    return results, best_model_name, best_model, preprocessor


def train_classification_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate 7 classification models + check imbalance techniques."""
    print("\n--- Training Classification Models (High Demand Prediction) ---")
    
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 1. Compare Imbalance Resampling Techniques
    # Note: When using median split, the dataset is balanced (50/50).
    # To demonstrate SMOTE, ADASYN, and ROS per the syllabus, we evaluate them on a peak-demand imbalanced target (Top 25% demand)
    print("  Evaluating Class Imbalance Handling Techniques (Baseline, ROS, SMOTE, ADASYN):")
    p75 = np.percentile(X_train['Loc_Avg_Demand'], 75)
    # Create an imbalanced target for demonstration
    y_train_imb = (y_train == 1) & (X_train['Duration_Mins'] > X_train['Duration_Mins'].median())
    y_test_imb = (y_test == 1) & (X_test['Duration_Mins'] > X_test['Duration_Mins'].median())
    
    # Subset for fast resampling benchmark
    sub_idx = np.random.RandomState(42).choice(len(X_train_proc), size=min(15000, len(X_train_proc)), replace=False)
    X_sub = X_train_proc[sub_idx]
    y_sub = y_train_imb.values[sub_idx]
    
    imbalance_results = {}
    methods = {
        'Baseline (No Resampling)': (X_sub, y_sub),
        'Random Oversampling': RandomOverSampler(random_state=42).fit_resample(X_sub, y_sub),
        'SMOTE': SMOTE(random_state=42).fit_resample(X_sub, y_sub),
        'ADASYN': ADASYN(random_state=42).fit_resample(X_sub, y_sub)
    }

    for res_name, (X_res, y_res) in methods.items():
        lr_check = LogisticRegression(max_iter=1000, random_state=42)
        lr_check.fit(X_res, y_res)
        y_pred = lr_check.predict(X_test_proc)
        y_prob = lr_check.predict_proba(X_test_proc)[:, 1]
        imbalance_results[res_name] = {
            'Accuracy': round(float(accuracy_score(y_test_imb, y_pred)), 4),
            'Precision': round(float(precision_score(y_test_imb, y_pred, zero_division=0)), 4),
            'Recall': round(float(recall_score(y_test_imb, y_pred, zero_division=0)), 4),
            'F1': round(float(f1_score(y_test_imb, y_pred, zero_division=0)), 4),
            'ROC_AUC': round(float(roc_auc_score(y_test_imb, y_prob)), 4)
        }
        print(f"    {res_name} -> F1: {imbalance_results[res_name]['F1']}, Recall: {imbalance_results[res_name]['Recall']}, ROC_AUC: {imbalance_results[res_name]['ROC_AUC']}")

    # 2. Train and benchmark the required classifiers on the primary target
    linear_svc_cal = CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000, dual='auto'))

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        'SVM': linear_svc_cal,
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    }

    results = {}
    best_model_name = None
    best_f1 = -float('inf')
    best_model = None

    for name, model in models.items():
        print(f"  Training {name}...")
        if name == 'KNN':
            # Train KNN on stratified sample of 15,000 for responsive nearest-neighbor inference
            knn_idx = np.random.RandomState(42).choice(len(X_train_proc), size=min(15000, len(X_train_proc)), replace=False)
            model.fit(X_train_proc[knn_idx], y_train.values[knn_idx])
        else:
            model.fit(X_train_proc, y_train)
        preds = model.predict(X_test_proc)
        probs = model.predict_proba(X_test_proc)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, probs)

        results[name] = {
            'Accuracy': round(float(acc), 4),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'F1': round(float(f1), 4),
            'ROC_AUC': round(float(roc), 4)
        }
        print(f"    {name} -> Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.4f}, ROC_AUC: {roc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"Best Classification Model: {best_model_name} (F1 = {best_f1:.4f})")
    return results, imbalance_results, best_model_name, best_model


def run_time_series_analysis(df: pd.DataFrame):
    """
    Time-Series Analysis & Forecasting per requirements:
    - Aggregate demand by Daily intervals
    - Check stationarity via Augmented Dickey-Fuller (ADF)
    - Moving Average (7-day rolling)
    - Exponential Smoothing (Simple & Holt-Winters)
    - Evaluate using MAE & RMSE
    """
    print("\n--- Running Time-Series Forecasting Analysis ---")
    
    # Resample daily demand
    daily_ts = df.set_index('StartDatetime').resample('D')['Demand'].sum().dropna()
    print(f"  Daily Time-Series length: {len(daily_ts)} days (from {daily_ts.index.min().strftime('%Y-%m-%d')} to {daily_ts.index.max().strftime('%Y-%m-%d')})")

    # 1. Stationarity Test (ADF)
    adf_result = adfuller(daily_ts.values)
    adf_stat, p_val = float(adf_result[0]), float(adf_result[1])
    is_stationary = bool(p_val < 0.05)
    print(f"  ADF Statistic: {adf_stat:.4f}, p-value: {p_val:.4e} -> Stationary: {is_stationary}")

    # Train/Test Split (Last 30 days as test set)
    test_days = min(30, int(len(daily_ts) * 0.2))
    train_ts = daily_ts.iloc[:-test_days]
    test_ts = daily_ts.iloc[-test_days:]

    # 2. Moving Average (7-day rolling average forecast)
    ma_preds = [train_ts.iloc[-7:].mean()] * len(test_ts)
    ma_mae = mean_absolute_error(test_ts, ma_preds)
    ma_rmse = np.sqrt(mean_squared_error(test_ts, ma_preds))

    # 3. Simple Exponential Smoothing
    ses_model = SimpleExpSmoothing(train_ts, initialization_method='estimated').fit()
    ses_preds = ses_model.forecast(len(test_ts))
    ses_mae = mean_absolute_error(test_ts, ses_preds)
    ses_rmse = np.sqrt(mean_squared_error(test_ts, ses_preds))

    # 4. Holt-Winters Exponential Smoothing (with additive trend and seasonality if enough periods)
    try:
        hw_model = ExponentialSmoothing(train_ts, trend='add', seasonal='add', seasonal_periods=7, initialization_method='estimated').fit()
        hw_preds = hw_model.forecast(len(test_ts))
        hw_mae = mean_absolute_error(test_ts, hw_preds)
        hw_rmse = np.sqrt(mean_squared_error(test_ts, hw_preds))
    except Exception as e:
        print(f"  Holt-Winters seasonal fallback to Holt's linear: {e}")
        hw_model = ExponentialSmoothing(train_ts, trend='add', initialization_method='estimated').fit()
        hw_preds = hw_model.forecast(len(test_ts))
        hw_mae = mean_absolute_error(test_ts, hw_preds)
        hw_rmse = np.sqrt(mean_squared_error(test_ts, hw_preds))

    ts_results = {
        'Stationarity_ADF': {
            'ADF_Statistic': round(adf_stat, 4),
            'p_value': round(p_val, 6),
            'Is_Stationary': is_stationary
        },
        'Models': {
            'Moving Average (7-Day)': {'MAE': round(float(ma_mae), 2), 'RMSE': round(float(ma_rmse), 2)},
            'Simple Exponential Smoothing': {'MAE': round(float(ses_mae), 2), 'RMSE': round(float(ses_rmse), 2)},
            'Holt-Winters Exponential Smoothing': {'MAE': round(float(hw_mae), 2), 'RMSE': round(float(hw_rmse), 2)}
        }
    }

    print("  Time Series Forecast Performance:")
    for m_name, metrics in ts_results['Models'].items():
        print(f"    {m_name} -> MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}")

    # Save daily aggregated time series for dashboard rendering
    daily_ts_df = daily_ts.reset_index()
    daily_ts_df.columns = ['Date', 'Total_Demand_kWh']
    daily_ts_df['Date'] = daily_ts_df['Date'].dt.strftime('%Y-%m-%d')

    return ts_results, daily_ts_df, hw_model


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_charging_records.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading preprocessed data from: {data_path}")
    df = load_processed_data(data_path)

    # Prepare features
    X, y_reg, y_clf, preprocessor, cat_cols, num_cols = prepare_features(df)
    
    # Train/Test Split (80/20)
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.20, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.20, random_state=42)

    # 1. Train Regression Models
    reg_results, best_reg_name, best_reg_model, fitted_preprocessor = train_regression_models(
        X_train, X_test, y_reg_train, y_reg_test, preprocessor
    )

    # 2. Train Classification Models
    clf_results, imbalance_results, best_clf_name, best_clf_model = train_classification_models(
        X_train, X_test, y_clf_train, y_clf_test, fitted_preprocessor
    )

    # 3. Time Series Forecasting
    ts_results, daily_ts_df, hw_model = run_time_series_analysis(df)

    # 4. Save Models and Metadata
    print("\n--- Serializing Models and Metadata ---")
    joblib.dump(best_reg_model, os.path.join(models_dir, 'regression_model.pkl'))
    joblib.dump(best_clf_model, os.path.join(models_dir, 'classification_model.pkl'))
    joblib.dump(fitted_preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
    joblib.dump(hw_model, os.path.join(models_dir, 'timeseries_model.pkl'))

    # Save processed sample for SHAP/LIME fast dashboard explanations
    sample_df = df.sample(n=min(1500, len(df)), random_state=42)
    sample_df.to_csv(os.path.join(models_dir, 'shap_sample_data.csv'), index=False)
    daily_ts_df.to_csv(os.path.join(models_dir, 'daily_timeseries.csv'), index=False)

    metadata = {
        'best_regression_model': best_reg_name,
        'best_classification_model': best_clf_name,
        'regression_metrics': reg_results,
        'classification_metrics': clf_results,
        'imbalance_comparison': imbalance_results,
        'timeseries_metrics': ts_results,
        'categorical_columns': cat_cols,
        'numerical_columns': num_cols,
        'demand_median_threshold': float(df['Demand'].median()),
        'total_records_trained': len(df)
    }

    with open(os.path.join(models_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print("All models and metrics successfully saved to:", models_dir)
    print("\nPipeline execution complete!")


if __name__ == '__main__':
    main()
