import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import time
import pandas as pd
from src.preprocessing.real_sensor_prep import preprocess_real_sensor_dataframe
from src.features.vibration_features import extract_vibration_features
from src.features.audio_features import extract_audio_features
from src.features.temperature_features import extract_temperature_features
from src.inference.inference import RetroFitInferencePipeline
from src.alerts.escalation import AlertEscalationEngine
from src.communication.mqtt_client import RetroFitMQTTPublisher

def run_replay_demo(csv_path=r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv", models_dir="data/models", replay_speed=1.0):
    print("=========================================================")
    print("       RETROFIT REPLAY / DEMO MODE (RECORDED DATA)      ")
    print("  [LABEL: DEMO / RECORDED DATA REPLAY - NOT LIVE DATA]   ")
    print("=========================================================")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Replay CSV file {csv_path} not found.")
        return
        
    print(f"[REPLAY] Loading raw sensor recording from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    
    pipeline = RetroFitInferencePipeline(models_dir=models_dir)
    escalator = AlertEscalationEngine(confirmation_count=3, min_confidence=70.0, cooldown_period_sec=300)
    publisher = RetroFitMQTTPublisher()
    
    machine_config = {
        'machine_id': pipeline.fingerprint.get('machine_id', 'LAPTOP_IDLE_01'),
        'machine_name': 'Demo Laptop',
        'location': 'Testing Lab',
        'operator_phone': '+15550199',
        'alert_policy': {'critical_confirmation_count': 3, 'min_confidence': 70.0, 'cooldown_period_sec': 300}
    }
    
    # Feature extraction on recorded windows
    features_csv = "data/features/real_features.csv"
    if not os.path.exists(features_csv):
        print("[REPLAY] Extracting features from replay dataset...")
        from scripts.extract_real_features import run_real_feature_extraction
        df_features = run_real_feature_extraction(csv_path, "config.yaml", features_csv)
    else:
        df_features = pd.read_csv(features_csv)
        
    print(f"\n[REPLAY STREAMING] Replaying {len(df_features)} window frames through ML Pipeline...")
    
    for idx in range(len(df_features)):
        feat_win = df_features.iloc[[idx]]
        
        # 1. Inference
        out = pipeline.predict_window(feat_win)
        
        # 2. Escalation & Twilio Engine
        alert_action = escalator.evaluate_observation(
            status=out['status'],
            anomaly_score=out['anomaly_score'],
            drift_score=out['behavior_drift'],
            confidence=out['confidence'],
            machine_config=machine_config,
            top_features=out.get('top_features', [])
        )
        
        # 3. Publish to MQTT Subscriber
        pub_res = publisher.publish_telemetry(out, machine_config=machine_config)
        
        print(f"[DEMO REPLAY Frame {idx+1:2d}/{len(df_features)}] "
              f"Status: {out['status']:23s} | "
              f"Similarity: {out['similarity']:5.1f}% | "
              f"Drift: {out['behavior_drift']:4.1f} | "
              f"Alert: {alert_action['alert_level']:8s}")
              
        if replay_speed > 0:
            time.sleep(0.05 / replay_speed)
            
    print("\n=========================================================")
    print("          REPLAY / DEMO STREAMING COMPLETE SUCCESS!      ")
    print("=========================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RetroFit Demo Replay Mode")
    parser.add_argument("--csv", default=r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv")
    parser.add_argument("--models", default="data/models")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier (e.g. 1.0, 2.0, 5.0)")
    args = parser.parse_args()
    
    run_replay_demo(csv_path=args.csv, models_dir=args.models, replay_speed=args.speed)
