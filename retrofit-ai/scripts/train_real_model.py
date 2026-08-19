import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import yaml
import pandas as pd

from scripts.extract_real_features import run_real_feature_extraction
from src.training.cross_session_pipeline import run_cross_session_pipeline

def train_real_pipeline(csv_path=r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv", config_path="config.yaml", train_sessions=None, test_sessions=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("=========================================================")
    print("   RETROFIT LEAK-FREE REAL ML TRAINING & PIPELINE        ")
    print("=========================================================")
    
    # 1. Extract Features
    real_features_csv = "data/features/real_features.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Real data CSV {csv_path} not found.")
        return
        
    df_features = run_real_feature_extraction(csv_path, config_path, real_features_csv)
    
    # 2. Run Cross-Session Leak-Free Pipeline
    res = run_cross_session_pipeline(
        df_features=df_features,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
        top_n=30,
        output_dir="data/models"
    )
    
    print("\n=========================================================")
    print("   LEAK-FREE REAL TRAINING PIPELINE COMPLETE!            ")
    print("=========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train_real_pipeline(args.csv, args.config)
