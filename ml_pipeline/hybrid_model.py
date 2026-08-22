"""
RetroFit Hybrid Machine Behavioral Intelligence Engine
Combines Martin's State Discovery (GMM) with Rushwa's Neural Latent Fingerprinting (Autoencoder),
providing state-aware anomaly detection, multimodal root-cause explainability, and safe adaptive baseline tracking.
"""

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

def extract_features_from_window(df_win):
    """Calculates all 52 features for a window."""
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
    return [feats[col] for col in FEATURE_COLUMNS]

class RetroFitHybridPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=2, random_state=42)
        self.autoencoder = None
        self.encoder = None
        self.state_fingerprints = {}
        self.state_thresholds = {}
        self.global_threshold = 2.5
        
    def fit(self, df_features, epochs=80, batch_size=32):
        feature_cols = [c for c in df_features.columns if c not in ['window_id', 'start_timestamp_ms', 'end_timestamp_ms']]
        X_raw = df_features[feature_cols].values
        
        # 1. Fit Scaler
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # 2. Fit Operating State Model (GMM)
        self.gmm.fit(X_scaled)
        states = self.gmm.predict(X_scaled)
        
        # 3. Build & Train Autoencoder
        inputs = layers.Input(shape=(len(feature_cols),), name='input_features')
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dense(16, activation='relu')(x)
        latent = layers.Dense(8, activation='linear', name='machine_fingerprint')(x)
        y = layers.Dense(16, activation='relu')(latent)
        y = layers.Dense(32, activation='relu')(y)
        outputs = layers.Dense(len(feature_cols), activation='linear', name='reconstruction')(y)
        
        self.autoencoder = keras.Model(inputs=inputs, outputs=outputs, name='RetroFit_Autoencoder')
        self.encoder = keras.Model(inputs=inputs, outputs=latent, name='RetroFit_Fingerprint_Encoder')
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
        
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=0)
        ]
        self.autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
        
        # 4. Extract State-Specific Fingerprints & Thresholds
        latents = self.encoder.predict(X_scaled, verbose=0)
        recons = self.autoencoder.predict(X_scaled, verbose=0)
        recon_errors = np.mean((X_scaled - recons)**2, axis=1)
        
        for k in range(self.gmm.n_components):
            mask_k = (states == k)
            if np.sum(mask_k) >= 5:
                state_fp = np.mean(latents[mask_k], axis=0).tolist()
                fp_dists_k = np.linalg.norm(latents[mask_k] - np.array(state_fp), axis=1)
                recon_k = recon_errors[mask_k]
                
                # Combined score: 0.5 * normalized recon + 0.5 * normalized distance
                norm_r = (recon_k - np.mean(recon_k)) / (np.std(recon_k) + 1e-8)
                norm_d = (fp_dists_k - np.mean(fp_dists_k)) / (np.std(fp_dists_k) + 1e-8)
                scores_k = 0.5 * np.maximum(0, norm_r) + 0.5 * np.maximum(0, norm_d)
                
                self.state_fingerprints[int(k)] = state_fp
                self.state_thresholds[int(k)] = float(np.percentile(scores_k, 99.0))
            else:
                self.state_fingerprints[int(k)] = np.mean(latents, axis=0).tolist()
                self.state_thresholds[int(k)] = 2.5
                
        self.global_threshold = float(np.mean(list(self.state_thresholds.values())))
        return self

    def predict_window(self, feature_vector_52):
        """Runs fast inference on a single 52-feature vector."""
        x = np.array(feature_vector_52, dtype=float).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        
        # State Discovery
        state = int(self.gmm.predict(x_scaled)[0])
        state_fp = np.array(self.state_fingerprints.get(state, self.state_fingerprints[0]))
        thresh = self.state_thresholds.get(state, self.global_threshold)
        
        # Latent Fingerprint
        latent = self.encoder.predict(x_scaled, verbose=0)[0]
        fp_dist = float(np.linalg.norm(latent - state_fp))
        
        # Reconstruction & MSE
        recon = self.autoencoder.predict(x_scaled, verbose=0)[0]
        feat_errs = (x_scaled[0] - recon)**2
        recon_err = float(np.mean(feat_errs))
        
        # Score & Similarity
        anomaly_score = float(0.5 * (recon_err / (0.05 + 1e-8)) + 0.5 * (fp_dist / (1.5 + 1e-8)))
        similarity_score = float(np.clip(100.0 * (1.0 - (anomaly_score / (thresh * 2.0 + 1e-8))), 0.0, 100.0))
        status = "ANOMALY" if anomaly_score > thresh else "HEALTHY"
        
        # Root-Cause Breakdown
        vib_idx = [FEATURE_COLUMNS.index(c) for c in MODALITY_MAP['Vibration_Motion']]
        ac_idx  = [FEATURE_COLUMNS.index(c) for c in MODALITY_MAP['Acoustic']]
        th_idx  = [FEATURE_COLUMNS.index(c) for c in MODALITY_MAP['Thermal']]
        
        total_err = float(np.sum(feat_errs) + 1e-8)
        vib_pct = round(float(np.sum(feat_errs[vib_idx]) / total_err * 100.0), 1)
        ac_pct  = round(float(np.sum(feat_errs[ac_idx])  / total_err * 100.0), 1)
        th_pct  = round(float(np.sum(feat_errs[th_idx])  / total_err * 100.0), 1)
        
        top_idx = np.argsort(feat_errs)[::-1][:3]
        top_features = [f"{FEATURE_COLUMNS[i]} ({round(float(feat_errs[i]/total_err*100), 1)}%)" for i in top_idx]
        
        return {
            'state': state,
            'status': status,
            'similarity_score': round(similarity_score, 1),
            'anomaly_score': round(anomaly_score, 3),
            'threshold': round(thresh, 3),
            'latent_fingerprint': [round(float(v), 3) for v in latent],
            'modality_breakdown': {
                'vibration_pct': vib_pct,
                'acoustic_pct': ac_pct,
                'thermal_pct': th_pct
            },
            'top_contributing_causes': top_features
        }

    def save(self, model_dir, physical_baselines=None):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
        joblib.dump(self.gmm, os.path.join(model_dir, "gmm_state_model.pkl"))
        self.autoencoder.save(os.path.join(model_dir, "retrofit_autoencoder.keras"))
        self.encoder.save(os.path.join(model_dir, "retrofit_encoder.keras"))
        
        # Export TFLite
        conv = tf.lite.TFLiteConverter.from_keras_model(self.autoencoder)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = conv.convert()
        with open(os.path.join(model_dir, "retrofit_autoencoder.tflite"), "wb") as f:
            f.write(tflite_bytes)
            
        conv_enc = tf.lite.TFLiteConverter.from_keras_model(self.encoder)
        conv_enc.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_enc_bytes = conv_enc.convert()
        with open(os.path.join(model_dir, "retrofit_encoder.tflite"), "wb") as f:
            f.write(tflite_enc_bytes)
            
        params = {
            'state_fingerprints': self.state_fingerprints,
            'state_thresholds': self.state_thresholds,
            'global_threshold': self.global_threshold,
            'feature_columns': FEATURE_COLUMNS,
            'physical_baselines': physical_baselines or {}
        }
        with open(os.path.join(model_dir, "hybrid_parameters.json"), "w") as f:
            json.dump(params, f, indent=2)
        with open(os.path.join(model_dir, "hybrid_parameters.pkl"), "wb") as f:
            pickle.dump(params, f)
        print(f"[EXPORT] Hybrid RetroFit Model Artifacts saved to {model_dir}")
