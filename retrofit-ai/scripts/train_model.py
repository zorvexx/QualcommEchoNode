import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.extract_features import run_feature_extraction
from src.selection.selector import select_top_features
from src.states.state_discovery import OperatingStateDiscoverer
from src.models.model_comparison import compare_anomaly_models
from src.behavior.fingerprint import MachineFingerprint
from src.training.prepare_dataset import session_aware_split
from src.training.export_model import export_model_artifacts

def train_pipeline(data_path, config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("=========================================================")
    print("       RETROFIT MACHINE BEHAVIORAL AI PIPELINE           ")
    print("=========================================================")
    
    # 1. Extract Features
    features_csv = config['paths']['features_csv']
    if not os.path.exists(data_path):
        print(f"[ERROR] Raw data path {data_path} not found.")
        return
        
    df_features = run_feature_extraction(data_path, config_path, features_csv)
    
    # 2. Separate Metadata & Target
    meta_cols = ['timestamp', 'machine_id', 'session_id', 'operating_state', 'label']
    feature_cols = [c for c in df_features.columns if c not in meta_cols]
    
    X = df_features[feature_cols]
    y = df_features['label'].values if 'label' in df_features.columns else np.zeros(len(df_features))
    
    # Filter healthy data for training
    healthy_mask = (y == 0)
    X_healthy = X[healthy_mask]
    
    # 3. Feature Selection
    print(f"\n[STEP 1] Performing Data-Driven Feature Selection (Candidate Features: {len(feature_cols)})...")
    top_n = config['selection']['top_n_features']
    selected_features, ranking_df = select_top_features(X_healthy, y[healthy_mask] if len(np.unique(y[healthy_mask])) > 1 else None, top_n=top_n)
    
    os.makedirs(os.path.dirname(config['paths']['feature_importance_csv']), exist_ok=True)
    ranking_df.to_csv(config['paths']['feature_importance_csv'], index=False)
    print(f" -> Selected Top {len(selected_features)} Features: {selected_features[:5]}...")
    
    # 4. Session-Aware Split
    X_selected = df_features[selected_features]
    df_selected_full = df_features[meta_cols + selected_features]
    train_df, val_df, test_df = session_aware_split(df_selected_full, val_ratio=0.2, test_ratio=0.2)
    
    X_train_h = train_df[train_df['label'] == 0][selected_features]
    X_test = test_df[selected_features]
    y_test = test_df['label'].values
    
    # 5. Scaler Fitting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_h)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Operating State Discovery
    print(f"\n[STEP 2] Operating State Discovery (PCA + GMM Clustering)...")
    state_discoverer = OperatingStateDiscoverer(
        method='gmm',
        min_clusters=config['clustering']['min_clusters'],
        max_clusters=config['clustering']['max_clusters']
    ).fit(X_train_scaled)
    
    print(f" -> Discovered {state_discoverer.best_k} Optimal Operating States.")
    
    # 7. Model Comparison & Training
    print(f"\n[STEP 3] Benchmarking Anomaly Detection Candidates...")
    df_comp, best_name, best_model, best_thresh = compare_anomaly_models(X_train_scaled, X_test_scaled, y_test)
    
    comp_path = os.path.join(config['paths']['models_dir'], "model_comparison.csv")
    os.makedirs(os.path.dirname(comp_path), exist_ok=True)
    df_comp.to_csv(comp_path, index=False)
    print("\n--- MODEL COMPARISON METRICS ---")
    print(df_comp.to_string(index=False))
    print(f"\n >>> WINNING MODEL SELECTED: {best_name} (Threshold: {best_thresh:.4f})")
    
    # 8. Compute State-Specific Anomaly Thresholds
    train_states = state_discoverer.predict(X_train_scaled)
    train_scores = best_model.predict_score(X_train_scaled)
    state_thresholds = {}
    for k in range(state_discoverer.best_k):
        mask_k = (train_states == k)
        if np.sum(mask_k) >= 5:
            scores_k = train_scores[mask_k]
            state_thresh_k = max(float(best_thresh), float(np.percentile(scores_k, 99)))
            state_thresholds[str(k)] = state_thresh_k
        else:
            state_thresholds[str(k)] = float(best_thresh)
    print(f" -> State-Specific Anomaly Thresholds (Lower-bounded by Global 99th Pct): {state_thresholds}")

    # 9. Machine Behavioral Fingerprint
    print(f"\n[STEP 4] Constructing Machine Behavioral Fingerprint...")
    fp_builder = MachineFingerprint(machine_id=df_features['machine_id'].iloc[0])
    fp_data = fp_builder.build_fingerprint(
        X_healthy=df_features[healthy_mask],
        selected_features=selected_features,
        state_model=state_discoverer,
        anomaly_threshold=best_thresh,
        state_thresholds=state_thresholds
    )
    
    # 9. Export Artifacts
    print(f"\n[STEP 5] Exporting Edge & Training Model Artifacts...")
    export_model_artifacts(best_model, scaler, selected_features, state_discoverer, fp_data, output_dir=config['paths']['models_dir'])
    
    print("\n=========================================================")
    print("       TRAINING & BEHAVIORAL MODELING COMPLETE!          ")
    print("=========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/demo_machine.csv")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train_pipeline(args.data, args.config)
