"""
RetroFit Autoencoder Training & Behavioral Fingerprint Engine
Builds, trains, and exports the neural autoencoder, latent fingerprint extractor,
and statistical anomaly thresholds.
"""

import os
import pickle
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_autoencoder(input_dim=52, latent_dim=8):
    """
    Constructs the symmetric Dense Autoencoder matching Martin's architecture.
    Encoder: 52 -> 32 -> 16 -> 8 (latent fingerprint)
    Decoder: 8 -> 16 -> 32 -> 52 (reconstruction)
    """
    inputs = layers.Input(shape=(input_dim,), name='input_features')
    
    # Encoder
    x = layers.Dense(32, activation='relu', name='enc_dense1')(inputs)
    x = layers.Dense(16, activation='relu', name='enc_dense2')(x)
    latent = layers.Dense(latent_dim, activation='linear', name='machine_fingerprint')(x)
    
    # Decoder
    y = layers.Dense(16, activation='relu', name='dec_dense1')(latent)
    y = layers.Dense(32, activation='relu', name='dec_dense2')(y)
    outputs = layers.Dense(input_dim, activation='linear', name='reconstruction')(y)
    
    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name='RetroFit_Autoencoder')
    encoder = keras.Model(inputs=inputs, outputs=latent, name='RetroFit_Fingerprint_Encoder')
    
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return autoencoder, encoder

def train_model(features_df, output_dir, epochs=100, batch_size=32, val_split=0.2, alpha=0.5, beta=0.5):
    """
    Trains the model on healthy baseline features and computes the normal fingerprint.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load 52 features
    feature_cols = [c for c in features_df.columns if c not in ['window_id', 'start_timestamp_ms', 'end_timestamp_ms']]
    X_raw = features_df[feature_cols].values
    
    print(f"Training on {len(X_raw)} windows with {len(feature_cols)} features...")
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Train / validation split
    X_train, X_val = train_test_split(X_scaled, test_size=val_split, random_state=42, shuffle=True)
    
    autoencoder, encoder = build_autoencoder(input_dim=len(feature_cols), latent_dim=8)
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=1)
    ]
    
    print("\nStarting Autoencoder training...")
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Compute normal baseline fingerprint (centroid of latent space on healthy training data)
    latent_train = encoder.predict(X_train, verbose=0)
    normal_fingerprint = np.mean(latent_train, axis=0).tolist()
    
    # Compute training baseline reconstruction errors and fingerprint distances
    train_recon = autoencoder.predict(X_train, verbose=0)
    train_recon_errors = np.mean((X_train - train_recon)**2, axis=1)
    train_fp_distances = np.linalg.norm(latent_train - np.array(normal_fingerprint), axis=1)
    
    train_recon_mean = float(np.mean(train_recon_errors))
    train_recon_std  = float(np.std(train_recon_errors))
    train_fp_mean    = float(np.mean(train_fp_distances))
    train_fp_std     = float(np.std(train_fp_distances))
    
    # Compute normalized anomaly scores on training set to set statistical threshold
    norm_recon = (train_recon_errors - train_recon_mean) / (train_recon_std + 1e-8)
    norm_fp    = (train_fp_distances - train_fp_mean) / (train_fp_std + 1e-8)
    
    # Bound normalized scores >= 0
    norm_recon_clipped = np.maximum(0, norm_recon)
    norm_fp_clipped    = np.maximum(0, norm_fp)
    
    train_anomaly_scores = alpha * norm_recon_clipped + beta * norm_fp_clipped
    
    # Anomaly threshold: 99th percentile of healthy training anomaly score
    threshold = float(np.percentile(train_anomaly_scores, 99.0))
    
    params = {
        'alpha': alpha,
        'beta': beta,
        'threshold': threshold,
        'train_recon_mean': train_recon_mean,
        'train_recon_std': train_recon_std,
        'train_fp_mean': train_fp_mean,
        'train_fp_std': train_fp_std,
        'normal_fingerprint': normal_fingerprint,
        'feature_names': feature_cols
    }
    
    # Save artifacts
    ae_path = os.path.join(output_dir, "retrofit_autoencoder.keras")
    enc_path = os.path.join(output_dir, "retrofit_encoder.keras")
    scaler_path = os.path.join(output_dir, "retrofit_scaler.pkl")
    params_path = os.path.join(output_dir, "retrofit_anomaly_parameters.pkl")
    json_params_path = os.path.join(output_dir, "retrofit_anomaly_parameters.json")
    
    autoencoder.save(ae_path)
    encoder.save(enc_path)
    joblib.dump(scaler, scaler_path)
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    with open(json_params_path, "w") as f:
        json.dump(params, f, indent=2)
        
    print("\n[SUCCESS] Model training complete!")
    print(f"  -> Model saved: {ae_path}")
    print(f"  -> Encoder saved: {enc_path}")
    print(f"  -> Calibrated Anomaly Threshold: {threshold:.4f}")
    print(f"  -> 8-dim Baseline Fingerprint: {[round(v, 3) for v in normal_fingerprint]}")
    
    return params
