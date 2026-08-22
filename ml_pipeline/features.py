"""
RetroFit / RetroFit Feature Extraction Engine
Extracts 52 multimodal statistical and physical features over sliding windows
from synchronized 14-column sensor data (IMU, Acoustic, Thermal).
"""

import numpy as np
import pandas as pd
import os

FEATURE_COLUMNS = [
    # Accelerometer (15)
    'ax_mean', 'ax_std', 'ax_rms', 'ax_peak', 'ax_ptp',
    'ay_mean', 'ay_std', 'ay_rms', 'ay_peak', 'ay_ptp',
    'az_mean', 'az_std', 'az_rms', 'az_peak', 'az_ptp',
    # Acceleration Magnitude (5)
    'acc_mag_mean', 'acc_mag_std', 'acc_mag_rms', 'acc_mag_peak', 'acc_mag_ptp',
    # Gyroscope (12)
    'gx_mean', 'gx_std', 'gx_rms', 'gx_peak',
    'gy_mean', 'gy_std', 'gy_rms', 'gy_peak',
    'gz_mean', 'gz_std', 'gz_rms', 'gz_peak',
    # Gyroscope Magnitude (4)
    'gyro_mag_mean', 'gyro_mag_std', 'gyro_mag_rms', 'gyro_mag_peak',
    # Acoustic (8)
    'sound_peak_mean', 'sound_peak_std', 'sound_peak_rms', 'sound_peak_peak',
    'sound_volts_mean', 'sound_volts_std', 'sound_volts_rms', 'sound_volts_peak',
    # Thermal (8)
    'ir_object_mean', 'ir_object_std', 'ir_object_min', 'ir_object_max',
    'ir_object_range', 'temperature_slope',
    'ir_ambient_mean', 'ir_ambient_std'
]

# Modality mapping for Root-Cause Explainability
MODALITY_MAP = {
    'Vibration_Motion': [
        'ax_mean', 'ax_std', 'ax_rms', 'ax_peak', 'ax_ptp',
        'ay_mean', 'ay_std', 'ay_rms', 'ay_peak', 'ay_ptp',
        'az_mean', 'az_std', 'az_rms', 'az_peak', 'az_ptp',
        'acc_mag_mean', 'acc_mag_std', 'acc_mag_rms', 'acc_mag_peak', 'acc_mag_ptp',
        'gx_mean', 'gx_std', 'gx_rms', 'gx_peak',
        'gy_mean', 'gy_std', 'gy_rms', 'gy_peak',
        'gz_mean', 'gz_std', 'gz_rms', 'gz_peak',
        'gyro_mag_mean', 'gyro_mag_std', 'gyro_mag_rms', 'gyro_mag_peak'
    ],
    'Acoustic': [
        'sound_peak_mean', 'sound_peak_std', 'sound_peak_rms', 'sound_peak_peak',
        'sound_volts_mean', 'sound_volts_std', 'sound_volts_rms', 'sound_volts_peak'
    ],
    'Thermal': [
        'ir_object_mean', 'ir_object_std', 'ir_object_min', 'ir_object_max',
        'ir_object_range', 'temperature_slope',
        'ir_ambient_mean', 'ir_ambient_std'
    ]
}

def extract_window_features(df_win):
    """Computes the 52 features for a single sliding window DataFrame."""
    ax = df_win['ax'].values.astype(float)
    ay = df_win['ay'].values.astype(float)
    az = df_win['az'].values.astype(float)
    
    gx = df_win['gx'].values.astype(float)
    gy = df_win['gy'].values.astype(float)
    gz = df_win['gz'].values.astype(float)
    
    sound_peak = df_win['sound_peak'].values.astype(float)
    sound_volts = df_win['sound_volts'].values.astype(float)
    
    ir_obj = df_win['ir_object_c'].values.astype(float)
    ir_amb = df_win['ir_ambient_c'].values.astype(float)
    ts_ms = df_win['timestamp_ms'].values.astype(float)
    
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    dt_sec = (ts_ms[-1] - ts_ms[0]) / 1000.0 if len(ts_ms) > 1 and (ts_ms[-1] - ts_ms[0]) > 0 else 1.0
    temp_slope = (ir_obj[-1] - ir_obj[0]) / dt_sec if dt_sec > 0 else 0.0
    
    feats = {
        'ax_mean': np.mean(ax), 'ax_std': np.std(ax), 'ax_rms': np.sqrt(np.mean(ax**2)), 'ax_peak': np.max(np.abs(ax)), 'ax_ptp': np.ptp(ax),
        'ay_mean': np.mean(ay), 'ay_std': np.std(ay), 'ay_rms': np.sqrt(np.mean(ay**2)), 'ay_peak': np.max(np.abs(ay)), 'ay_ptp': np.ptp(ay),
        'az_mean': np.mean(az), 'az_std': np.std(az), 'az_rms': np.sqrt(np.mean(az**2)), 'az_peak': np.max(np.abs(az)), 'az_ptp': np.ptp(az),
        
        'acc_mag_mean': np.mean(acc_mag), 'acc_mag_std': np.std(acc_mag), 'acc_mag_rms': np.sqrt(np.mean(acc_mag**2)), 'acc_mag_peak': np.max(acc_mag), 'acc_mag_ptp': np.ptp(acc_mag),
        
        'gx_mean': np.mean(gx), 'gx_std': np.std(gx), 'gx_rms': np.sqrt(np.mean(gx**2)), 'gx_peak': np.max(np.abs(gx)),
        'gy_mean': np.mean(gy), 'gy_std': np.std(gy), 'gy_rms': np.sqrt(np.mean(gy**2)), 'gy_peak': np.max(np.abs(gy)),
        'gz_mean': np.mean(gz), 'gz_std': np.std(gz), 'gz_rms': np.sqrt(np.mean(gz**2)), 'gz_peak': np.max(np.abs(gz)),
        
        'gyro_mag_mean': np.mean(gyro_mag), 'gyro_mag_std': np.std(gyro_mag), 'gyro_mag_rms': np.sqrt(np.mean(gyro_mag**2)), 'gyro_mag_peak': np.max(gyro_mag),
        
        'sound_peak_mean': np.mean(sound_peak), 'sound_peak_std': np.std(sound_peak), 'sound_peak_rms': np.sqrt(np.mean(sound_peak**2)), 'sound_peak_peak': np.max(sound_peak),
        'sound_volts_mean': np.mean(sound_volts), 'sound_volts_std': np.std(sound_volts), 'sound_volts_rms': np.sqrt(np.mean(sound_volts**2)), 'sound_volts_peak': np.max(sound_volts),
        
        'ir_object_mean': np.mean(ir_obj), 'ir_object_std': np.std(ir_obj), 'ir_object_min': np.min(ir_obj), 'ir_object_max': np.max(ir_obj),
        'ir_object_range': np.ptp(ir_obj), 'temperature_slope': temp_slope,
        'ir_ambient_mean': np.mean(ir_amb), 'ir_ambient_std': np.std(ir_amb)
    }
    return feats

def extract_dataset_features(csv_path, window_size=400, step_size=200, output_csv=None):
    """
    Extracts features across the full dataset using sliding windows.
    window_size = 400 samples (~5.7s @ 70Hz)
    step_size   = 200 samples (~2.8s overlap)
    """
    df = pd.read_csv(csv_path)
    n_samples = len(df)
    
    # Advanced Acoustic Signal Processing Filter:
    # 5-sample rolling median (rejects transient noise clicks) + EMA smoothing (preserves continuous operating sound)
    if 'sound_volts' in df.columns:
        df['sound_volts'] = df['sound_volts'].rolling(5, min_periods=1, center=True).median().ewm(span=3).mean()
    if 'sound_peak' in df.columns:
        df['sound_peak'] = df['sound_peak'].rolling(5, min_periods=1, center=True).median().ewm(span=3).mean()
        
    print(f"Extracting features from {csv_path} ({n_samples} rows)...")
    
    rows = []
    win_id = 0
    for start_idx in range(0, n_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        df_win = df.iloc[start_idx:end_idx]
        
        start_ts = df_win['timestamp_ms'].iloc[0]
        end_ts = df_win['timestamp_ms'].iloc[-1]
        
        feats = extract_window_features(df_win)
        feats['window_id'] = win_id
        feats['start_timestamp_ms'] = start_ts
        feats['end_timestamp_ms'] = end_ts
        
        rows.append(feats)
        win_id += 1
        
    df_features = pd.DataFrame(rows)
    cols = ['window_id', 'start_timestamp_ms', 'end_timestamp_ms'] + FEATURE_COLUMNS
    df_features = df_features[cols]
    
    if output_csv:
        df_features.to_csv(output_csv, index=False)
        print(f"[SUCCESS] Extracted {len(df_features)} windows -> {output_csv}")
        
    return df_features
