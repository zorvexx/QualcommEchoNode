"""
RetroFit Anomaly Inference & Root-Cause Explainability Engine
Computes Reconstruction Error, Fingerprint Distance, Anomaly Score, Similarity %,
and breaks down root causes by modality (Vibration %, Acoustic %, Thermal %).
"""

import os
import pickle
import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Modality grouping
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

class RetroFitInferEngine:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.autoencoder = keras.models.load_model(os.path.join(model_dir, "echonode_autoencoder.keras"))
        self.encoder = keras.models.load_model(os.path.join(model_dir, "echonode_encoder.keras"))
        self.scaler = joblib.load(os.path.join(model_dir, "echonode_scaler.pkl"))
        
        with open(os.path.join(model_dir, "echonode_anomaly_parameters.pkl"), "rb") as f:
            self.params = pickle.load(f)
            
        self.feature_names = self.params.get('feature_names', list(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else None)
        self.normal_fingerprint = np.array(self.params['normal_fingerprint'])
        self.threshold = self.params['threshold']
        self.alpha = self.params['alpha']
        self.beta = self.params['beta']
        
    def evaluate_features(self, df_features, output_results_csv=None):
        feature_cols = [c for c in df_features.columns if c not in ['window_id', 'start_timestamp_ms', 'end_timestamp_ms']]
        X_raw = df_features[feature_cols].values
        X_scaled = self.scaler.transform(X_raw)
        
        # 1. Latent fingerprint & distance
        latent = self.encoder.predict(X_scaled, verbose=0)
        fp_distances = np.linalg.norm(latent - self.normal_fingerprint, axis=1)
        
        # 2. Reconstruction & error
        recon = self.autoencoder.predict(X_scaled, verbose=0)
        recon_errors = np.mean((X_scaled - recon)**2, axis=1)
        
        # 3. Per-feature squared errors (for explainability)
        feat_errors = (X_scaled - recon)**2
        
        # 4. Normalized scores
        norm_recon = np.maximum(0, (recon_errors - self.params['train_recon_mean']) / (self.params['train_recon_std'] + 1e-8))
        norm_fp    = np.maximum(0, (fp_distances - self.params['train_fp_mean']) / (self.params['train_fp_std'] + 1e-8))
        
        anomaly_scores = self.alpha * norm_recon + self.beta * norm_fp
        similarity_scores = np.clip(100.0 * (1.0 - (anomaly_scores / (self.threshold * 2.0 + 1e-8))), 0.0, 100.0)
        
        # 5. Modality breakdown & top contributors
        vibration_indices = [feature_cols.index(c) for c in MODALITY_MAP['Vibration_Motion'] if c in feature_cols]
        acoustic_indices  = [feature_cols.index(c) for c in MODALITY_MAP['Acoustic'] if c in feature_cols]
        thermal_indices   = [feature_cols.index(c) for c in MODALITY_MAP['Thermal'] if c in feature_cols]
        
        results = []
        for i in range(len(df_features)):
            total_err = np.sum(feat_errors[i]) + 1e-8
            vib_err = np.sum(feat_errors[i, vibration_indices])
            ac_err  = np.sum(feat_errors[i, acoustic_indices])
            th_err  = np.sum(feat_errors[i, thermal_indices])
            
            vib_pct = round(float((vib_err / total_err) * 100.0), 1)
            ac_pct  = round(float((ac_err / total_err) * 100.0), 1)
            th_pct  = round(float((th_err / total_err) * 100.0), 1)
            
            # Top 3 contributing features
            top_3_idx = np.argsort(feat_errors[i])[::-1][:3]
            top_3_str = ", ".join([f"{feature_cols[idx]} ({round(float(feat_errors[i, idx]/total_err*100), 1)}%)" for idx in top_3_idx])
            
            score = float(anomaly_scores[i])
            status = "ANOMALY" if score > self.threshold else "HEALTHY"
            
            res_row = {
                'window_id': df_features['window_id'].iloc[i] if 'window_id' in df_features else i,
                'start_timestamp_ms': df_features['start_timestamp_ms'].iloc[i] if 'start_timestamp_ms' in df_features else 0,
                'reconstruction_error': float(recon_errors[i]),
                'fingerprint_distance': float(fp_distances[i]),
                'anomaly_score': score,
                'similarity_score': float(similarity_scores[i]),
                'status': status,
                'vibration_pct': vib_pct,
                'acoustic_pct': ac_pct,
                'thermal_pct': th_pct,
                'top_contributing_features': top_3_str
            }
            results.append(res_row)
            
        df_results = pd.DataFrame(results)
        
        if output_results_csv:
            df_results.to_csv(output_results_csv, index=False)
            print(f"[SUCCESS] Exported results & explainability -> {output_results_csv}")
            
        return df_results
