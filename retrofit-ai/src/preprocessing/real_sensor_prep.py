import os
import json
import numpy as np
import pandas as pd

def load_or_compute_gyro_calibration(df_raw=None, cal_file="data/calibration/gyro_calibration.json", max_noise_std_threshold=500.0):
    """
    Loads zero-rate offsets from dedicated stationary calibration recording or computes them.
    Includes noise/stability check and exports quality metrics to data/calibration/calibration_quality.json.
    """
    if os.path.exists(cal_file):
        with open(cal_file, 'r') as f:
            cal = json.load(f)
            print(f"[CALIBRATION] Loaded dedicated stationary gyro offsets from {cal_file}: {cal}")
            return cal
            
    if df_raw is not None and 'gx' in df_raw.columns:
        zro_samples = min(500, len(df_raw))
        gx_s = df_raw['gx'].iloc[:zro_samples]
        gy_s = df_raw['gy'].iloc[:zro_samples]
        gz_s = df_raw['gz'].iloc[:zro_samples]
        
        std_gx = float(np.std(gx_s))
        std_gy = float(np.std(gy_s))
        std_gz = float(np.std(gz_s))
        
        is_stable = bool(std_gx < max_noise_std_threshold and std_gy < max_noise_std_threshold and std_gz < max_noise_std_threshold)
        
        if not is_stable:
            print(f"[CALIBRATION WARNING] High noise/movement detected during calibration! (Std gx:{std_gx:.1f}, gy:{std_gy:.1f}, gz:{std_gz:.1f} > threshold {max_noise_std_threshold}). Check sensor stability.")
        else:
            print(f"[CALIBRATION PASSED] Calibration noise within bounds (Std gx:{std_gx:.1f}, gy:{std_gy:.1f}, gz:{std_gz:.1f} < {max_noise_std_threshold}).")
            
        cal = {
            'zro_gx': float(np.mean(gx_s)),
            'zro_gy': float(np.mean(gy_s)),
            'zro_gz': float(np.mean(gz_s)),
            'source': 'dataset_initial_samples'
        }
        
        quality = {
            'std_gx': round(std_gx, 2),
            'std_gy': round(std_gy, 2),
            'std_gz': round(std_gz, 2),
            'max_noise_std_threshold': max_noise_std_threshold,
            'is_stable': is_stable
        }
        
        os.makedirs(os.path.dirname(cal_file), exist_ok=True)
        with open(cal_file, 'w') as f:
            json.dump(cal, f, indent=2)
            
        quality_file = os.path.join(os.path.dirname(cal_file), "calibration_quality.json")
        with open(quality_file, 'w') as f:
            json.dump(quality, f, indent=2)
            
        return cal
        
    return {'zro_gx': 0.0, 'zro_gy': 0.0, 'zro_gz': 0.0, 'source': 'default_zero'}

def preprocess_real_sensor_dataframe(df_raw, calibrate_gyro=True, cal_file="data/calibration/gyro_calibration.json"):
    """
    Preprocesses raw real sensor data:
    1. Removes magnetometer channels (mx, my, mz)
    2. Subtracts Zero-Rate Offset (ZRO) using dedicated calibration file
    3. Calculates vector magnitudes: acc_mag and gyro_mag
    """
    df = df_raw.copy()
    
    # 1. Drop Magnetometer Channels if present
    mag_cols = [c for c in df.columns if c in ['mx', 'my', 'mz']]
    if mag_cols:
        df.drop(columns=mag_cols, inplace=True)
        
    # 2. Vector Magnitudes
    if 'ax' in df.columns and 'ay' in df.columns and 'az' in df.columns:
        df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        
    if 'gx' in df.columns and 'gy' in df.columns and 'gz' in df.columns:
        if calibrate_gyro:
            cal = load_or_compute_gyro_calibration(df_raw, cal_file=cal_file)
            df['gx_cal'] = df['gx'] - cal['zro_gx']
            df['gy_cal'] = df['gy'] - cal['zro_gy']
            df['gz_cal'] = df['gz'] - cal['zro_gz']
            df['gyro_mag'] = np.sqrt(df['gx_cal']**2 + df['gy_cal']**2 + df['gz_cal']**2)
        else:
            df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
            
    return df
