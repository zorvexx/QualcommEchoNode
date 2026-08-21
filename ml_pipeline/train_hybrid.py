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

def train_and_export_hybrid(custom_data_path=None):
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = custom_data_path or os.path.join(workspace, "mlx90614_dataset_converted.csv")
    model_dir = os.path.join(workspace, "ml_pipeline", "models")
    
    print("=" * 65)
    print("   TRAINING RETROFIT HYBRID MODEL (GMM + AUTOENCODER)   ")
    print(f"   Target Dataset: {data_path}")
    print("=" * 65)
    
    # 1. Compute Physical Baselines of this specific machine
    import numpy as np
    df_raw = pd.read_csv(data_path)
    ax_g = df_raw['ax'] / 16384.0
    ay_g = df_raw['ay'] / 16384.0
    az_g = df_raw['az'] / 16384.0
    acc_mag = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)
    vib_std_val = float(np.std(acc_mag))
    sound_mean = float(df_raw['sound_volts'].mean())
    temp_delta_mean = float((df_raw['ir_object_c'] - df_raw['ir_ambient_c']).mean())
    
    physical_baselines = {
        "vib_std_mean": round(vib_std_val, 4),
        "vib_std_min": round(max(0.001, vib_std_val * 0.35), 4),
        "vib_std_max": round(max(0.025, vib_std_val * 2.20), 4),
        "sound_energy_mean": round(sound_mean, 4),
        "sound_energy_min": round(max(0.02, sound_mean * 0.30), 4),
        "sound_energy_max": round(max(0.50, sound_mean * 2.50), 4),
        "temp_delta_mean": round(temp_delta_mean, 2),
        "temp_delta_min": round(temp_delta_mean - 3.5, 2),
        "temp_delta_max": round(temp_delta_mean + 3.5, 2)
    }
    print(f"\n[LEARNED PHYSICAL BASELINE PROFILE]")
    print(f"  - Vibration Std:   {physical_baselines['vib_std_mean']:.4f}g (Normal Band: {physical_baselines['vib_std_min']:.4f}g - {physical_baselines['vib_std_max']:.4f}g)")
    print(f"  - Sound AC Energy: {physical_baselines['sound_energy_mean']:.4f}V (Normal Band: {physical_baselines['sound_energy_min']:.4f}V - {physical_baselines['sound_energy_max']:.4f}V)")
    print(f"  - Temp Gradient:   {physical_baselines['temp_delta_mean']:+.2f}C (Normal Band: {physical_baselines['temp_delta_min']:+.2f}C to {physical_baselines['temp_delta_max']:+.2f}C)\n")
    
    # 2. Feature Extraction
    features_csv = os.path.join(model_dir, "hybrid_features.csv")
    os.makedirs(model_dir, exist_ok=True)
    df_features = extract_dataset_features(data_path, window_size=400, step_size=200, output_csv=features_csv)
    
    # 3. Train Hybrid Model
    print("\nFitting Hybrid State Discovery + Neural Fingerprint Engine...")
    pipeline = RetroFitHybridPipeline()
    pipeline.fit(df_features, epochs=80)
    
    # 4. Export Artifacts with Physical Baselines
    pipeline.save(model_dir, physical_baselines=physical_baselines)
    
    # Also save to root models folder
    root_model_dir = os.path.join(workspace, "models")
    pipeline.save(root_model_dir, physical_baselines=physical_baselines)
    
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
    target = sys.argv[1] if len(sys.argv) > 1 else None
    train_and_export_hybrid(target)
