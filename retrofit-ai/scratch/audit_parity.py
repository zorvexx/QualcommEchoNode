import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
import pandas as pd

from src.preprocessing.real_sensor_prep import load_or_compute_gyro_calibration, preprocess_real_sensor_dataframe
from src.features.vibration_features import extract_vibration_features
from src.features.audio_features import extract_audio_features
from src.features.temperature_features import extract_temperature_features
from src.inference.inference import RetroFitInferencePipeline
from src.edge_sim.edge_runner import run_uno_q_edge_simulation

def audit_parity():
    csv_path = r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv"
    
    # 1. Load trained selected features
    with open("data/models/selected_features.json", "r") as f:
        selected_features = json.load(f)
        
    print(f"Top 30 Selected Features: {selected_features}")
    
    # 2. Extract first window (410 samples) directly using reference python logic
    df_raw = pd.read_csv(csv_path)
    win_raw = df_raw.iloc[:410]
    
    win_clean = preprocess_real_sensor_dataframe(win_raw, calibrate_gyro=True)
    
    accel_data = win_clean[['ax', 'ay', 'az']].values
    gyro_data = win_clean[['gx_cal', 'gy_cal', 'gz_cal']].values
    
    f_vib = extract_vibration_features(accel_data, gyro_data, fs_accel=200, fs_gyro=200)
    f_aud = extract_audio_features(win_clean['sound_volts'].values, fs_audio=200, is_amplitude_summary=True)
    f_tmp = extract_temperature_features(win_clean['ir_object_c'].values, baseline_temp=float(win_clean['ir_ambient_c'].iloc[0]))
    
    ref_dict = {}
    ref_dict.update(f_vib)
    ref_dict.update(f_aud)
    ref_dict.update(f_tmp)
    
    # 3. Compare with Edge Simulator first window
    sim_res = run_uno_q_edge_simulation(csv_path=csv_path, machine_id="DEV_01")
    sim_output = sim_res['outputs'][0]
    
    # 4. Pipeline Direct Prediction
    direct_pipeline = RetroFitInferencePipeline(machine_id="DEV_01")
    
    df_ref_win = pd.DataFrame([ref_dict])
    for f in selected_features:
        if f not in df_ref_win.columns:
            df_ref_win[f] = 0.0
            
    direct_out = direct_pipeline.predict_window(df_ref_win)
    
    print("\n=========================================================")
    print("      PARITY CHECK: DIRECT PIPELINE VS EDGE SIMULATOR    ")
    print("=========================================================")
    print(f"Direct Pipeline Status : {direct_out['status']}")
    print(f"Edge Simulator Status  : {sim_output['status']}")
    print(f"Direct Anomaly Score   : {direct_out['anomaly_score']:.6f}")
    print(f"Edge Sim Anomaly Score : {sim_output['anomaly_score']:.6f}")
    print(f"Direct Similarity      : {direct_out['similarity']:.1f}%")
    print(f"Edge Sim Similarity    : {sim_output['similarity']:.1f}%")
    print("=========================================================")

if __name__ == "__main__":
    audit_parity()
