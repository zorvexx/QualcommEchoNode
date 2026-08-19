"""
Train Hybrid RetroFit Intelligence Model
Trains GMM State Discovery + Neural Autoencoder Fingerprinting on 1-Hour Real Dataset.
"""

import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_dataset_features
from hybrid_model import RetroFitHybridPipeline

def train_and_export_hybrid():
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(workspace, "mlx90614_dataset_converted.csv")
    model_dir = os.path.join(workspace, "ml_pipeline", "models")
    
    print("=" * 65)
    print("   TRAINING RETROFIT HYBRID MODEL (GMM + AUTOENCODER)   ")
    print("=" * 65)
    
    # 1. Feature Extraction
    features_csv = os.path.join(model_dir, "hybrid_features.csv")
    os.makedirs(model_dir, exist_ok=True)
    df_features = extract_dataset_features(data_path, window_size=400, step_size=200, output_csv=features_csv)
    
    # 2. Train Hybrid Model
    print("\nFitting Hybrid State Discovery + Neural Fingerprint Engine...")
    pipeline = RetroFitHybridPipeline()
    pipeline.fit(df_features, epochs=80)
    
    # 3. Export Artifacts
    pipeline.save(model_dir)
    
    # Also save to root models folder
    root_model_dir = os.path.join(workspace, "models")
    pipeline.save(root_model_dir)
    
    # 4. Test Single Window Inference
    print("\n--- SAMPLE LIVE EDGE INFERENCE TEST ---")
    sample_win = df_features.tail(1)
    feature_cols = [c for c in df_features.columns if c not in ['window_id', 'start_timestamp_ms', 'end_timestamp_ms']]
    sample_vec = sample_win[feature_cols].values[0]
    
    res = pipeline.predict_window(sample_vec)
    import json
    print(json.dumps(res, indent=2))
    print("\n[SUCCESS] Hybrid Model Trained, Tested, and Exported Successfully!")

if __name__ == "__main__":
    train_and_export_hybrid()
