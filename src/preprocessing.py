"""
Data Preprocessing & Feature Engineering Module
EV Charging Demand Prediction and Usage Pattern Analysis
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw EV charging records CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_charging_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    - Drop duplicates
    - Parse datetime formats
    - Calculate and validate duration
    - Handle anomalies / invalid records
    """
    df_clean = df.copy()

    # 1. Remove duplicate records
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_dupes = initial_count - len(df_clean)
    print(f"[Data Cleaning] Removed {removed_dupes} duplicate rows.")

    # 2. Convert datetimes
    df_clean['StartDatetime'] = pd.to_datetime(df_clean['StartDatetime'], errors='coerce')
    df_clean['EndDatetime'] = pd.to_datetime(df_clean['EndDatetime'], errors='coerce')

    # Drop any unparseable datetimes if present
    df_clean = df_clean.dropna(subset=['StartDatetime', 'EndDatetime'])

    # 3. Calculate true duration in minutes
    calc_duration = (df_clean['EndDatetime'] - df_clean['StartDatetime']).dt.total_seconds() / 60.0
    
    # If original Duration is invalid or <= 0, use calculated duration
    df_clean['Duration_Calculated'] = calc_duration
    df_clean['Duration_Mins'] = np.where(df_clean['Duration'] > 0, df_clean['Duration'], calc_duration)

    # 4. Filter invalid durations (<= 0 or extreme anomalies > 24 hours / 1440 mins)
    valid_mask = (df_clean['Duration_Mins'] > 0) & (df_clean['Duration_Mins'] <= 1440) & (df_clean['Demand'] > 0)
    filtered_out = len(df_clean) - valid_mask.sum()
    df_clean = df_clean[valid_mask].reset_index(drop=True)
    print(f"[Data Cleaning] Filtered {filtered_out} invalid/outlier duration/demand records.")
    print(f"[Data Cleaning] Cleaned dataset records: {len(df_clean):,}")

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer temporal, behavioral, and classification target features.
    """
    df_feat = df.copy()

    # 1. Temporal feature extraction
    df_feat['StartHour'] = df_feat['StartDatetime'].dt.hour
    df_feat['StartDay'] = df_feat['StartDatetime'].dt.day
    df_feat['StartMonth'] = df_feat['StartDatetime'].dt.month
    df_feat['DayOfWeek'] = df_feat['StartDatetime'].dt.day_of_week  # 0=Monday, 6=Sunday
    df_feat['DayName'] = df_feat['StartDatetime'].dt.day_name()
    df_feat['IsWeekend'] = df_feat['DayOfWeek'].isin([5, 6]).astype(int)

    # 2. Time of day binning
    def assign_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'

    df_feat['TimeOfDay'] = df_feat['StartHour'].apply(assign_time_of_day)

    # 3. Cyclical hour encoding (useful for neural nets & distance-based models)
    df_feat['Hour_Sin'] = np.sin(2 * np.pi * df_feat['StartHour'] / 24.0)
    df_feat['Hour_Cos'] = np.cos(2 * np.pi * df_feat['StartHour'] / 24.0)

    # 4. Historical / Usage aggregates
    loc_mean_demand = df_feat.groupby('Location')['Demand'].transform('mean')
    df_feat['Loc_Avg_Demand'] = loc_mean_demand

    charger_mean_demand = df_feat.groupby('ChargerType')['Demand'].transform('mean')
    df_feat['ChargerType_Avg_Demand'] = charger_mean_demand

    # 5. Charging Rate proxy (Demand / (Duration in hours))
    duration_hours = df_feat['Duration_Mins'] / 60.0
    df_feat['Avg_Power_kW'] = df_feat['Demand'] / duration_hours

    # 6. Classification Target: High Demand vs Normal Demand
    # Based on median demand as specified in the project requirements
    demand_median = float(df_feat['Demand'].median())
    df_feat['Demand_Median_Threshold'] = demand_median
    df_feat['High_Demand'] = (df_feat['Demand'] >= demand_median).astype(int)

    print(f"[Feature Engineering] Median Demand threshold: {demand_median:.2f} kWh")
    print(f"[Feature Engineering] High Demand count: {(df_feat['High_Demand'] == 1).sum():,} ({(df_feat['High_Demand'] == 1).mean():.1%})")
    print(f"[Feature Engineering] Normal Demand count: {(df_feat['High_Demand'] == 0).sum():,} ({(df_feat['High_Demand'] == 0).mean():.1%})")

    return df_feat


def save_pickled_dataset(data, pkl_path: str, metadata: dict = None) -> str:
    """
    Save a DataFrame or payload dictionary to disk as a serialized pickle (.pkl) artifact.
    """
    os.makedirs(os.path.dirname(os.path.abspath(pkl_path)), exist_ok=True)
    payload = {
        'timestamp': datetime.now().isoformat(),
        'data': data,
        'metadata': metadata or {}
    } if not isinstance(data, dict) or 'data' not in data else data
    
    joblib.dump(payload, pkl_path)
    size_kb = round(os.path.getsize(pkl_path) / 1024, 2)
    print(f"[Auto-Pickling] Successfully serialized artifact to: {pkl_path} ({size_kb} KB)")
    return pkl_path


def load_pickled_dataset(pkl_path: str):
    """
    Load a serialized pickle (.pkl) artifact from disk.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle artifact not found at: {pkl_path}")
    payload = joblib.load(pkl_path)
    if isinstance(payload, dict) and 'data' in payload:
        return payload['data'], payload
    return payload, {}


def process_and_pickle_uploaded_dataset(file_obj_or_path, preprocessor, reg_model, clf_model, df_baseline, output_pickle_path: str = None):
    """
    End-to-end processing and automatic serialization of uploaded CSV records:
    1. Reads and deduplicates records
    2. Validates datetimes and durations
    3. Extracts temporal and cyclic features
    4. Computes baseline historical features
    5. Runs batch predictions (Demand in kWh and High-Demand tier)
    6. Automatically serializes/pickles the scored dataset and feature matrix to disk
    """
    if isinstance(file_obj_or_path, str):
        df_up = pd.read_csv(file_obj_or_path)
        source_name = os.path.basename(file_obj_or_path)
    else:
        df_up = pd.read_csv(file_obj_or_path)
        source_name = getattr(file_obj_or_path, 'name', 'custom_uploaded.csv')

    df_up = df_up.drop_duplicates()

    # Parse datetimes
    df_up['StartDatetime'] = pd.to_datetime(df_up['StartDatetime'], errors='coerce')
    df_up['EndDatetime'] = pd.to_datetime(df_up['EndDatetime'], errors='coerce')
    df_up = df_up.dropna(subset=['StartDatetime', 'EndDatetime'])

    # Calculate duration
    calc_dur = (df_up['EndDatetime'] - df_up['StartDatetime']).dt.total_seconds() / 60.0
    if 'Duration' in df_up.columns:
        df_up['Duration_Mins'] = np.where(df_up['Duration'] > 0, df_up['Duration'], calc_dur)
    else:
        df_up['Duration_Mins'] = calc_dur

    valid_mask = (df_up['Duration_Mins'] > 0) & (df_up['Duration_Mins'] <= 1440)
    df_up = df_up[valid_mask].reset_index(drop=True)

    # Temporal feature extraction
    df_up['StartHour'] = df_up['StartDatetime'].dt.hour
    df_up['StartDay'] = df_up['StartDatetime'].dt.day
    df_up['StartMonth'] = df_up['StartDatetime'].dt.month
    df_up['DayOfWeek'] = df_up['StartDatetime'].dt.day_of_week
    df_up['DayName'] = df_up['StartDatetime'].dt.day_name()
    df_up['IsWeekend'] = df_up['DayOfWeek'].isin([5, 6]).astype(int)

    def assign_tod(h):
        if 5 <= h < 12: return 'Morning'
        elif 12 <= h < 17: return 'Afternoon'
        elif 17 <= h < 22: return 'Evening'
        else: return 'Night'
    df_up['TimeOfDay'] = df_up['StartHour'].apply(assign_tod)

    df_up['Hour_Sin'] = np.sin(2 * np.pi * df_up['StartHour'] / 24.0)
    df_up['Hour_Cos'] = np.cos(2 * np.pi * df_up['StartHour'] / 24.0)

    # Match historical baselines from training distribution
    loc_map = df_baseline.groupby('Location')['Demand'].mean().to_dict()
    type_map = df_baseline.groupby('ChargerType')['Demand'].mean().to_dict()
    overall_mean = float(df_baseline['Demand'].mean())

    df_up['Loc_Avg_Demand'] = df_up['Location'].map(loc_map).fillna(overall_mean)
    df_up['ChargerType_Avg_Demand'] = df_up['ChargerType'].map(type_map).fillna(overall_mean)

    feat_cols = ['Location', 'ChargerCompany', 'ChargerType', 'TimeOfDay', 'IsWeekend',
                 'Duration_Mins', 'StartHour', 'StartDay', 'StartMonth', 'DayOfWeek',
                 'Hour_Sin', 'Hour_Cos', 'Loc_Avg_Demand', 'ChargerType_Avg_Demand']

    # Batch ML Predictions
    X_proc = preprocessor.transform(df_up[feat_cols])
    df_up['Predicted_Demand_kWh'] = np.round(np.maximum(0.5, reg_model.predict(X_proc)), 2)
    df_up['Predicted_High_Demand'] = clf_model.predict(X_proc)
    df_up['High_Demand_Prob'] = np.round(clf_model.predict_proba(X_proc)[:, 1], 4)
    df_up['Demand_Tier'] = np.where(df_up['Predicted_High_Demand'] == 1, 'High Demand', 'Normal Demand')

    if 'Demand' not in df_up.columns:
        df_up['Demand'] = df_up['Predicted_Demand_kWh']
        df_up['High_Demand'] = df_up['Predicted_High_Demand']
    else:
        if 'High_Demand' not in df_up.columns:
            df_up['High_Demand'] = (df_up['Demand'] >= 14.2).astype(int)

    # Auto-Pickle Serialization
    if output_pickle_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_pickle_path = os.path.join(base_dir, 'data', 'processed', 'uploaded_scored_data.pkl')

    os.makedirs(os.path.dirname(os.path.abspath(output_pickle_path)), exist_ok=True)
    payload = {
        'source_filename': source_name,
        'serialization_time': datetime.now().isoformat(),
        'total_sessions': len(df_up),
        'feature_columns': feat_cols,
        'data': df_up,
        'predicted_total_demand_kwh': float(df_up['Predicted_Demand_kWh'].sum()),
        'predicted_avg_demand_kwh': float(df_up['Predicted_Demand_kWh'].mean()),
        'predicted_high_demand_ratio': float(df_up['Predicted_High_Demand'].mean()),
        'models_applied': {
            'preprocessor': 'preprocessor.pkl',
            'regression': 'regression_model.pkl',
            'classification': 'classification_model.pkl'
        }
    }
    joblib.dump(payload, output_pickle_path)
    size_kb = round(os.path.getsize(output_pickle_path) / 1024, 2)
    print(f"[Auto-Pickling] Successfully serialized uploaded dataset ({len(df_up):,} rows) to: {output_pickle_path} ({size_kb} KB)")

    pickle_info = {
        'pickle_path': output_pickle_path,
        'file_size_kb': size_kb,
        'serialization_time': payload['serialization_time'],
        'total_sessions': len(df_up),
        'payload': payload
    }

    return df_up, pickle_info


def run_pipeline(raw_path: str, output_path: str):
    """Run end-to-end preprocessing pipeline and save cleaned data as both CSV and Pickle."""
    print(f"Loading data from: {raw_path}")
    raw_df = load_raw_data(raw_path)
    clean_df = clean_charging_data(raw_df)
    final_df = engineer_features(clean_df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Cleaned and engineered dataset successfully saved to: {output_path}")

    # Also auto-pickle the cleaned dataset
    pkl_output = os.path.splitext(output_path)[0] + '.pkl'
    save_pickled_dataset(final_df, pkl_output, metadata={'source': raw_path, 'rows': len(final_df)})
    return final_df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, 'data', 'raw', 'ChargingRecords.csv')
    processed_csv = os.path.join(base_dir, 'data', 'processed', 'cleaned_charging_records.csv')
    run_pipeline(raw_csv, processed_csv)
