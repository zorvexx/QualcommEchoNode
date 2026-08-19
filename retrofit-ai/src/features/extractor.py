import numpy as np
import pandas as pd
from src.preprocessing.vibration import preprocess_vibration
from src.preprocessing.audio import preprocess_audio
from src.preprocessing.temperature import preprocess_temperature
from src.preprocessing.synchronization import synchronize_multimodal_df
from src.features.vibration_features import extract_vibration_features
from src.features.audio_features import extract_audio_features
from src.features.temperature_features import extract_temperature_features
from src.features.magnetometer_features import extract_magnetometer_features

def extract_features_from_dataframe(df, config):
    """
    Applies sliding window over dataframe and extracts full candidate feature set per window.
    """
    df = synchronize_multimodal_df(df)
    
    accel_fs = config['sampling']['accel_fs']
    gyro_fs = config['sampling']['gyro_fs']
    audio_fs = config['sampling']['audio_fs']
    temp_fs = config['sampling']['temp_fs']
    mag_fs = config['sampling']['mag_fs']
    
    win_sec = config['window']['duration_seconds']
    overlap = config['window']['overlap']
    hop_sec = win_sec * (1.0 - overlap)
    
    # Calculate samples per window for base accel sampling rate
    n_accel_win = int(accel_fs * win_sec)
    n_accel_hop = int(accel_fs * hop_sec)
    
    total_len = len(df)
    features_list = []
    
    start_idx = 0
    while start_idx + n_accel_win <= total_len:
        end_idx = start_idx + n_accel_win
        df_win = df.iloc[start_idx:end_idx]
        
        timestamp = df_win['timestamp'].iloc[-1] if 'timestamp' in df_win.columns else start_idx
        machine_id = df_win['machine_id'].iloc[-1] if 'machine_id' in df_win.columns else 'MACHINE_01'
        session_id = df_win['session_id'].iloc[-1] if 'session_id' in df_win.columns else 1
        op_state = df_win['operating_state'].iloc[-1] if 'operating_state' in df_win.columns else 'NORMAL'
        label = df_win['label'].iloc[-1] if 'label' in df_win.columns else 0
        
        # 1. Vibration
        accel_raw = df_win[['ax', 'ay', 'az']].values
        gyro_raw = df_win[['gx', 'gy', 'gz']].values
        accel_clean = preprocess_vibration(accel_raw, fs=accel_fs)
        vib_feats = extract_vibration_features(accel_clean, gyro_raw, fs_accel=accel_fs, fs_gyro=gyro_fs,
                                              enable_wavelets=config['features']['enable_wavelets'])
        
        # 2. Audio
        audio_raw = df_win['audio'].values
        audio_clean = preprocess_audio(audio_raw, fs=audio_fs)
        audio_feats = extract_audio_features(audio_clean, fs_audio=audio_fs,
                                             enable_mfcc=config['features']['enable_mfcc'])
        
        # 3. Temperature
        temp_raw = df_win['temperature'].values
        temp_clean = preprocess_temperature(temp_raw)
        temp_feats = extract_temperature_features(temp_clean)
        
        # 4. Magnetometer
        mag_feats = {}
        if config['features']['enable_magnetometer'] and 'mx' in df_win.columns and not df_win['mx'].isna().all():
            mag_raw = df_win[['mx', 'my', 'mz']].values
            mag_feats = extract_magnetometer_features(mag_raw, fs_mag=mag_fs)
            
        row = {
            'timestamp': timestamp,
            'machine_id': machine_id,
            'session_id': session_id,
            'operating_state': op_state,
            'label': label
        }
        row.update(vib_feats)
        row.update(audio_feats)
        row.update(temp_feats)
        row.update(mag_feats)
        
        features_list.append(row)
        start_idx += n_accel_hop
        
    return pd.DataFrame(features_list)
