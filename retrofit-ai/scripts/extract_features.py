import os
import argparse
import yaml
import pandas as pd
from src.features.extractor import extract_features_from_dataframe

def run_feature_extraction(data_path, config_path="config.yaml", output_path="data/features/features.csv"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"[EXTRACT] Reading raw sensor data from {data_path}...")
    df_raw = pd.read_csv(data_path)
    
    print(f"[EXTRACT] Running sliding window feature extraction...")
    df_features = extract_features_from_dataframe(df_raw, config)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"[EXTRACT] Extracted {len(df_features)} window feature vectors with {df_features.shape[1]} features -> {output_path}")
    return df_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/demo_machine.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/features/features.csv")
    args = parser.parse_args()
    run_feature_extraction(args.data, args.config, args.output)
