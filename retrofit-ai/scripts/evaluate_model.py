import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import pandas as pd
import numpy as np
from src.inference.inference import RetroFitInferencePipeline

def run_evaluation(data_path="data/raw/demo_machine.csv", models_dir="data/models"):
    print(f"[EVALUATE] Loading inference pipeline from {models_dir}...")
    pipeline = RetroFitInferencePipeline(models_dir=models_dir)
    
    print(f"[EVALUATE] Loading evaluation dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    
    from src.features.extractor import extract_features_from_dataframe
    import yaml
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    df_features = extract_features_from_dataframe(df_raw, config)
    
    results = []
    print(f"[EVALUATE] Executing inference loop over {len(df_features)} window feature vectors...")
    for idx, row in df_features.iterrows():
        df_row = pd.DataFrame([row])
        res = pipeline.predict_window(df_row)
        res['true_label'] = row.get('label', 0)
        results.append(res)
        
    df_res = pd.DataFrame(results)
    
    print("\n--- EVALUATION SAMPLE OUTPUT (LAST 5 WINDOWS) ---")
    cols_show = ['machine_id', 'state', 'similarity', 'behavior_drift', 'status', 'confidence', 'true_label']
    print(df_res[cols_show].tail(10).to_string(index=False))
    
    acc = np.mean((df_res['status'] != 'NORMAL_OPERATION').astype(int) == df_res['true_label'])
    print(f"\n[EVALUATION METRIC] Overall Classification Accuracy: {acc * 100:.2f}%")
    return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/demo_machine.csv")
    parser.add_argument("--models", default="data/models")
    args = parser.parse_args()
    run_evaluation(args.data, args.models)
