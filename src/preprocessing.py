"""
Data Preprocessing & Feature Engineering Module
EV Charging Demand Prediction and Usage Pattern Analysis
"""

import os
import pandas as pd
import numpy as np


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


def run_pipeline(raw_path: str, output_path: str):
    """Run end-to-end preprocessing pipeline and save cleaned data."""
    print(f"Loading data from: {raw_path}")
    raw_df = load_raw_data(raw_path)
    clean_df = clean_charging_data(raw_df)
    final_df = engineer_features(clean_df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Cleaned and engineered dataset successfully saved to: {output_path}")
    return final_df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, 'data', 'raw', 'ChargingRecords.csv')
    processed_csv = os.path.join(base_dir, 'data', 'processed', 'cleaned_charging_records.csv')
    run_pipeline(raw_csv, processed_csv)
