"""
Test Script: Auto-Pickling Verification for Uploaded EV Datasets
"""

import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import joblib
import pandas as pd
import numpy as np
from src.preprocessing import process_and_pickle_uploaded_dataset

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data')

    print("=" * 70)
    print("⚡ TESTING AUTO-PICKLING CAPABILITY FOR UPLOADED DATASETS")
    print("=" * 70)

    # 1. Load models
    print("\n[Step 1] Loading pickled ML pipelines...")
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
    reg_model = joblib.load(os.path.join(models_dir, 'regression_model.pkl'))
    clf_model = joblib.load(os.path.join(models_dir, 'classification_model.pkl'))
    clean_pkl = os.path.join(data_dir, 'processed', 'cleaned_charging_records.pkl')
    
    if os.path.exists(clean_pkl):
        df_baseline = joblib.load(clean_pkl)
        print(f"✓ Loaded baseline from pickled artifact: {len(df_baseline):,} records")
    else:
        df_baseline = pd.read_csv(os.path.join(data_dir, 'processed', 'cleaned_charging_records.csv'))
        print(f"✓ Loaded baseline from CSV: {len(df_baseline):,} records")

    # 2. Create sample test CSV for upload simulation
    print("\n[Step 2] Simulating client upload file with 100 sample sessions...")
    test_sample = df_baseline.sample(100, random_state=123)[
        ['StartDatetime', 'EndDatetime', 'Duration', 'Location', 'ChargerType', 'ChargerCompany']
    ]
    test_upload_csv = os.path.join(data_dir, 'processed', 'test_sample_upload.csv')
    test_sample.to_csv(test_upload_csv, index=False)
    print(f"✓ Created test upload CSV at: {test_upload_csv}")

    # 3. Execute auto-pickling processing
    print("\n[Step 3] Running process_and_pickle_uploaded_dataset...")
    target_pickle_path = os.path.join(data_dir, 'processed', 'uploaded_scored_data.pkl')
    
    # Remove old pickle if exists to ensure fresh generation
    if os.path.exists(target_pickle_path):
        os.remove(target_pickle_path)

    df_scored, pickle_info = process_and_pickle_uploaded_dataset(
        test_upload_csv, preprocessor, reg_model, clf_model, df_baseline, output_pickle_path=target_pickle_path
    )

    # 4. Verify auto-pickling artifact on disk
    print("\n[Step 4] Verifying generated pickle artifact on disk...")
    assert os.path.exists(target_pickle_path), f"ERROR: Pickle file was not created at {target_pickle_path}!"
    file_size_kb = os.path.getsize(target_pickle_path) / 1024
    print(f"✓ Pickle file verified: {target_pickle_path} ({file_size_kb:.2f} KB)")

    # 5. Deserialize and validate payload contents
    print("\n[Step 5] Deserializing via joblib.load and validating payload contents...")
    loaded_payload = joblib.load(target_pickle_path)
    assert 'data' in loaded_payload, "ERROR: 'data' key missing in serialized payload!"
    loaded_df = loaded_payload['data']
    
    assert len(loaded_df) == 100, f"ERROR: Expected 100 rows, got {len(loaded_df)}"
    assert 'Predicted_Demand_kWh' in loaded_df.columns, "ERROR: Predicted_Demand_kWh missing in loaded data!"
    assert 'Predicted_High_Demand' in loaded_df.columns, "ERROR: Predicted_High_Demand missing in loaded data!"
    assert 'High_Demand_Prob' in loaded_df.columns, "ERROR: High_Demand_Prob missing in loaded data!"

    print("✓ Row count validation: 100 / 100 sessions verified")
    print(f"✓ Mean predicted demand: {loaded_df['Predicted_Demand_kWh'].mean():.2f} kWh")
    print(f"✓ High-demand ratio: {(loaded_df['Predicted_High_Demand'].mean())*100:.1f}%")
    print(f"✓ Source filename in metadata: {loaded_payload.get('source_filename')}")
    print(f"✓ Serialization timestamp: {loaded_payload.get('serialization_time')}")

    print("\n" + "=" * 70)
    print("🎉 ALL AUTO-PICKLING CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    main()
