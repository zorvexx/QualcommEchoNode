import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import yaml
import numpy as np
import pandas as pd

from src.preprocessing.real_sensor_prep import preprocess_real_sensor_dataframe
from src.features.vibration_features import extract_vibration_features
from src.features.audio_features import extract_audio_features
from src.features.temperature_features import extract_temperature_features

def run_real_feature_extraction(csv_path=r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv", config_path="config.yaml", output_csv="data/features/real_features.csv"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"[EXTRACT REAL] Reading real raw dataset from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    
    # 1. Preprocess & Calibrate
    df_clean = preprocess_real_sensor_dataframe(df_raw, calibrate_gyro=True)
    
    # Check timestamp
    ts_col = 'timestamp_ms' if 'timestamp_ms' in df_clean.columns else df_clean.columns[0]
    ts = df_clean[ts_col].values.astype(float)
    
    win_sec = config['window']['duration_seconds'] # e.g. 2.048s
    overlap = config['window']['overlap'] # 0.5
    hop_sec = win_sec * (1.0 - overlap) # 1.024s
    
    # Estimate average nominal sampling rate
    dt_ms = np.diff(ts)
    dt_valid = dt_ms[dt_ms > 0]
    nominal_fs = 1000.0 / np.median(dt_valid) if len(dt_valid) > 0 else 200.0
    print(f" -> Nominal Baseline Sampling Rate: {nominal_fs:.1f} Hz (Median dt: {np.median(dt_valid):.2f} ms)")
    
    samples_per_win = int(nominal_fs * win_sec) # ~410 samples
    samples_per_hop = int(nominal_fs * hop_sec) # ~205 samples
    
    total_len = len(df_clean)
    features_list = []
    
    start_idx = 0
    win_count = 0
    while start_idx + samples_per_win <= total_len:
        end_idx = start_idx + samples_per_win
        df_win = df_clean.iloc[start_idx:end_idx]
        
        # Calculate dynamic window effective sampling frequency
        ts_win = df_win[ts_col].values.astype(float)
        dur_win_sec = (ts_win[-1] - ts_win[0]) / 1000.0
        eff_fs_win = (len(ts_win) - 1) / dur_win_sec if dur_win_sec > 0 else nominal_fs
        
        timestamp = ts_win[-1]
        machine_id = df_win['machine_id'].iloc[-1] if 'machine_id' in df_win.columns else 'LAPTOP_IDLE_01'
        session_label = df_win['session_label'].iloc[-1] if 'session_label' in df_win.columns else (df_win['session_id'].iloc[-1] if 'session_id' in df_win.columns else 'idle_01')
        op_state = df_win['operating_state'].iloc[-1] if 'operating_state' in df_win.columns else 'IDLE_NORMAL'
        label = df_win['label'].iloc[-1] if 'label' in df_win.columns else 0
        
        # 1. Vibration Features (Accel + Gyro)
        accel_raw = df_win[['ax', 'ay', 'az']].values
        gyro_cols = ['gx_cal', 'gy_cal', 'gz_cal'] if 'gx_cal' in df_win.columns else ['gx', 'gy', 'gz']
        gyro_raw = df_win[gyro_cols].values
        
        vib_feats = extract_vibration_features(accel_raw, gyro_raw, fs_accel=eff_fs_win, fs_gyro=eff_fs_win,
                                              enable_wavelets=config['features']['enable_wavelets'])
        
        # Add vector magnitude specific features
        acc_mag = df_win['acc_mag'].values
        gyro_mag = df_win['gyro_mag'].values
        vib_feats['acc_mag_mean'] = float(np.mean(acc_mag))
        vib_feats['acc_mag_std'] = float(np.std(acc_mag))
        vib_feats['acc_mag_max'] = float(np.max(acc_mag))
        vib_feats['acc_mag_min'] = float(np.min(acc_mag))
        vib_feats['acc_mag_rms'] = float(np.sqrt(np.mean(acc_mag**2)))
        
        vib_feats['gyro_mag_mean'] = float(np.mean(gyro_mag))
        vib_feats['gyro_mag_std'] = float(np.std(gyro_mag))
        vib_feats['gyro_mag_max'] = float(np.max(gyro_mag))
        vib_feats['gyro_mag_min'] = float(np.min(gyro_mag))
        vib_feats['gyro_mag_rms'] = float(np.sqrt(np.mean(gyro_mag**2)))
        
        # 2. Audio Features (MAX9814)
        audio_raw = df_win['sound_volts'].values if 'sound_volts' in df_win.columns else df_win['sound_peak'].values
        audio_feats = extract_audio_features(audio_raw, fs_audio=eff_fs_win, enable_mfcc=config['features']['enable_mfcc'])
        
        # Add sound peak specific features
        if 'sound_peak' in df_win.columns:
            sp = df_win['sound_peak'].values
            audio_feats['sound_peak_mean'] = float(np.mean(sp))
            audio_feats['sound_peak_max'] = float(np.max(sp))
            audio_feats['sound_peak_std'] = float(np.std(sp))
            
        # 3. Temperature Features (MLX90614)
        ir_obj = df_win['ir_object_c'].values if 'ir_object_c' in df_win.columns else df_win['temperature'].values
        temp_feats = extract_temperature_features(ir_obj)
        
        if 'ir_ambient_c' in df_win.columns:
            ir_amb = df_win['ir_ambient_c'].values
            temp_feats['temp_ambient_mean'] = float(np.mean(ir_amb))
            temp_feats['temp_ambient_max'] = float(np.max(ir_amb))
            temp_feats['temp_ambient_min'] = float(np.min(ir_amb))
            temp_feats['temp_delta_mean'] = float(np.mean(ir_obj - ir_amb))
            
        row = {
            'timestamp': timestamp,
            'machine_id': machine_id,
            'session_label': session_label,
            'operating_state': op_state,
            'label': label,
            'eff_fs_win': round(eff_fs_win, 2)
        }
        row.update(vib_feats)
        row.update(audio_feats)
        row.update(temp_feats)
        
        features_list.append(row)
        start_idx += samples_per_hop
        win_count += 1
        
    df_features = pd.DataFrame(features_list)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_features.to_csv(output_csv, index=False)
    print(f" -> Extracted {len(df_features)} window feature vectors with {df_features.shape[1] - 6} candidate features -> {output_csv}")
    return df_features

if __name__ == "__main__":
    run_real_feature_extraction()
