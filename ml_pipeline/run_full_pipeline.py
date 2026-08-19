"""
RetroFit Complete End-to-End ML Pipeline Runner
1. Extracts 52 features from 14-column synchronized sensor data
2. Trains Neural Autoencoder + Latent 8-dim Fingerprint
3. Calibrates Statistical Anomaly Thresholds & Scaler
4. Runs Inference with Multimodal Root-Cause Explainability (Vib %, Audio %, Temp %)
5. Tests Safe Adaptive Drift Tracking
6. Exports Edge-Ready TFLite and C++ models for Arduino Uno Q
"""

import os
import sys
import argparse

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_dataset_features
from train import train_model
from inference import RetroFitInferEngine
from adaptive import SafeAdaptiveFingerprint
from export_edge import export_tflite_models

def main():
    parser = argparse.ArgumentParser(description="RetroFit ML Pipeline Runner")
    parser.add_argument("--data", type=str, default=None, help="Path to synchronized 14-column dataset CSV")
    parser.add_argument("--model_dir", type=str, default="Autoencoder model", help="Directory to save/load models")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    args = parser.parse_args()
    
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Locate Dataset
    if args.data is None:
        default_data = os.path.join(workspace, "mlx90614_dataset_converted.csv")
        if not os.path.exists(default_data):
            default_data = os.path.join(workspace, "mlx90614_dataset.csv")
        data_path = default_data
    else:
        data_path = args.data
        
    model_dir = os.path.join(workspace, args.model_dir) if not os.path.isabs(args.model_dir) else args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 70)
    print("RETROFIT ML PIPELINE -- REAL MACHINE BEHAVIORAL FINGERPRINTING")
    print("=" * 70)
    print(f"Dataset: {data_path}")
    print(f"Model Directory: {model_dir}")
    
    # 2. Extract Features
    features_csv = os.path.join(model_dir, "echonode_features.csv")
    df_features = extract_dataset_features(data_path, window_size=400, step_size=200, output_csv=features_csv)
    
    # 3. Train Model (if requested or if models don't exist)
    ae_file = os.path.join(model_dir, "echonode_autoencoder.keras")
    if args.retrain or not os.path.exists(ae_file):
        print("\n--- PHASE 1: MODEL TRAINING & FINGERPRINT EXTRACTION ---")
        params = train_model(df_features, model_dir, epochs=args.epochs)
    else:
        print("\n--- PHASE 1: USING EXISTING TRAINED MODEL ARTIFACTS ---")
        
    # 4. Inference & Explainability
    print("\n--- PHASE 2: INFERENCE & MULTIMODAL EXPLAINABILITY ---")
    engine = RetroFitInferEngine(model_dir)
    results_csv = os.path.join(model_dir, "echonode_inference_results.csv")
    df_results = engine.evaluate_features(df_features, output_results_csv=results_csv)
    
    n_total = len(df_results)
    n_anom = sum(df_results['status'] == "ANOMALY")
    n_healthy = n_total - n_anom
    print(f"\nEvaluation Summary across {n_total} windows:")
    print(f"  [+] Healthy Windows: {n_healthy} ({n_healthy/n_total*100:.1f}%)")
    print(f"  [!] Anomaly/Deviation Windows: {n_anom} ({n_anom/n_total*100:.1f}%)")
    print(f"  [*] Mean Similarity Score: {df_results['similarity_score'].mean():.1f}%")
    
    # Show sample explainability for any anomalies
    anom_rows = df_results[df_results['status'] == "ANOMALY"]
    if not anom_rows.empty:
        print("\nSample Anomaly Root-Cause Breakdown:")
        for _, row in anom_rows.head(5).iterrows():
            print(f"  Window {int(row['window_id'])} (Score: {row['anomaly_score']:.3f}, Similarity: {row['similarity_score']:.1f}%):")
            print(f"    - Modalities: Vibration {row['vibration_pct']}% | Acoustic {row['acoustic_pct']}% | Thermal {row['thermal_pct']}%")
            print(f"    - Top Causes: {row['top_contributing_features']}")
            
    # 5. Safe Adaptive Drift Test
    print("\n--- PHASE 3: SAFE ADAPTIVE FINGERPRINT TRACKING ---")
    adaptive = SafeAdaptiveFingerprint(engine.normal_fingerprint, learning_rate=0.005)
    
    encoder = engine.encoder
    X_raw = df_features[[c for c in df_features.columns if c not in ['window_id', 'start_timestamp_ms', 'end_timestamp_ms']]].values
    X_scaled = engine.scaler.transform(X_raw)
    latents = encoder.predict(X_scaled[:50], verbose=0)
    
    for i in range(min(50, len(df_results))):
        is_anom = (df_results['status'].iloc[i] == "ANOMALY")
        res = adaptive.step(latents[i], is_anomaly=is_anom)
        
    print(f"  Adaptive tracker processed 50 windows:")
    print(f"  -> Total Safe Baseline Updates: {adaptive.update_count}")
    print(f"  -> Total Anomaly Freezes: {adaptive.frozen_count}")
    print(f"  -> Total Factory Drift: {res['drift_from_factory']:.4f}")
    
    # 6. Edge Model Export
    print("\n--- PHASE 4: EDGE MODEL EXPORT FOR ARDUINO UNO Q ---")
    export_tflite_models(model_dir)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] RETROFIT COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
    print("=" * 70)

if __name__ == "__main__":
    main()
