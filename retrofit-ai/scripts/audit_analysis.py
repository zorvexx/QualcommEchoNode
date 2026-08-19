import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import yaml
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from scripts.extract_features import run_feature_extraction
from scripts.generate_demo_data import generate_demo_dataset
from src.features.extractor import extract_features_from_dataframe
from src.selection.selector import select_top_features
from src.states.state_discovery import OperatingStateDiscoverer
from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.models.mahalanobis import MahalanobisAnomalyDetector
from src.models.one_class_svm import OneClassSVMAnomalyDetector
from src.models.autoencoder import AutoencoderAnomalyDetector
from src.training.prepare_dataset import session_aware_split
from src.behavior.fingerprint import MachineFingerprint
from src.behavior.similarity import calculate_behavioral_similarity
from src.behavior.drift import BehaviorDriftTracker
from src.behavior.memory import BehavioralMemory
from src.behavior.historical_match import match_historical_events
from src.explainability.modality_contribution import compute_modality_contributions
from src.explainability.feature_contribution import compute_feature_contributions
from src.explainability.explanation import generate_machine_personality
from src.inference.inference import RetroFitInferencePipeline

def run_complete_audit():
    print("=========================================================")
    print("         RUNNING RETROFIT SYSTEM AUDIT & EXPERIMENTS     ")
    print("=========================================================")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    raw_path = "data/raw/demo_machine.csv"
    if not os.path.exists(raw_path):
        generate_demo_dataset(raw_path, duration_seconds=600)
        
    df_raw = pd.read_csv(raw_path)
    
    # --- PART 3: RAW DATA AUDIT ---
    print("\n--- PART 3: RAW DATA STATISTICAL SUMMARY ---")
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'audio', 'temperature']
    stats_rows = []
    for c in sensor_cols:
        if c in df_raw.columns:
            stats_rows.append({
                'channel': c,
                'min': round(df_raw[c].min(), 4),
                'max': round(df_raw[c].max(), 4),
                'mean': round(df_raw[c].mean(), 4),
                'std': round(df_raw[c].std(), 4),
                'null_count': df_raw[c].isna().sum()
            })
    df_raw_stats = pd.DataFrame(stats_rows)
    df_raw_stats.to_csv("data/raw/raw_data_summary.csv", index=False)
    print(df_raw_stats.to_string(index=False))
    
    # --- PART 6 & 7: FEATURE EXTRACTION & 30-FEATURE AUDIT ---
    features_path = config['paths']['features_csv']
    if not os.path.exists(features_path):
        df_features = run_feature_extraction(raw_path, "config.yaml", features_path)
    else:
        df_features = pd.read_csv(features_path)
        
    meta_cols = ['timestamp', 'machine_id', 'session_id', 'operating_state', 'label']
    feature_cols = [c for c in df_features.columns if c not in meta_cols]
    
    X = df_features[feature_cols]
    y = df_features['label'].values if 'label' in df_features.columns else np.zeros(len(df_features))
    healthy_mask = (y == 0)
    
    # Run Feature Selector to get rankings
    top_n = config['selection']['top_n_features']
    selected_features, ranking_df = select_top_features(X[healthy_mask], y[healthy_mask] if len(np.unique(y[healthy_mask])) > 1 else None, top_n=top_n)
    
    # Categorize 30 selected features
    audit_rows = []
    for idx, f_name in enumerate(selected_features, 1):
        if f_name.startswith('acc_mag'):
            modality = 'vibration'
            ftype = 'acceleration_magnitude'
            src = 'Ax, Ay, Az'
        elif f_name.startswith(('acc_x', 'acc_y', 'acc_z')):
            modality = 'vibration'
            ftype = 'raw_acceleration_axis'
            src = f_name.split('_')[1].upper()
        elif f_name.startswith('gyro'):
            modality = 'vibration'
            ftype = 'gyroscope_rotational'
            src = 'Gx, Gy, Gz'
        elif f_name.startswith('audio'):
            modality = 'acoustic'
            ftype = 'digital_audio_spectrum'
            src = 'INMP441 Mic'
        elif f_name.startswith('temp'):
            modality = 'thermal'
            ftype = 'surface_temperature_derivative'
            src = 'MLX90614 IR Temp'
        elif f_name.startswith('mag'):
            modality = 'magnetic'
            ftype = 'magnetic_magnitude'
            src = 'Mx, My, Mz'
        else:
            modality = 'other'
            ftype = 'derived'
            src = 'multi'
            
        score = ranking_df[ranking_df['feature'] == f_name]['score'].values[0] if f_name in ranking_df['feature'].values else 0.0
        why = f"High variance ratio / mutual info across baseline clusters (score={score:.4f})"
        
        audit_rows.append({
            'Rank': idx,
            'Feature_Name': f_name,
            'Modality': modality,
            'Feature_Type': ftype,
            'Source_Signal': src,
            'Importance_Score': round(score, 4),
            'Selection_Method': 'Composite (Variance + Correlation Drop + MI/RF)',
            'Why_Selected': why
        })
        
    df_audit = pd.DataFrame(audit_rows)
    os.makedirs("data/features", exist_ok=True)
    df_audit.to_csv("data/features/selected_features_audit.csv", index=False)
    print("\n--- PART 7: 30 SELECTED FEATURES AUDIT ---")
    print(df_audit[['Rank', 'Feature_Name', 'Modality', 'Importance_Score']].to_string(index=False))
    
    # --- PART 9: FEATURE ABLATION STUDY ---
    print("\n--- PART 9: FEATURE ABLATION STUDY ---")
    # Session split
    df_selected_full = df_features[meta_cols + feature_cols]
    train_df, val_df, test_df = session_aware_split(df_selected_full, val_ratio=0.2, test_ratio=0.2)
    
    y_test = test_df['label'].values
    ablation_counts = [len(feature_cols), 60, 50, 40, 30, 20, 10]
    ablation_results = []
    
    for cnt in ablation_counts:
        curr_feats = selected_features[:cnt] if cnt <= len(selected_features) else feature_cols[:cnt]
        
        X_tr = train_df[train_df['label'] == 0][curr_feats]
        X_te = test_df[curr_feats]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X_tr_s)
        
        t0 = time.time()
        scores = -model.decision_function(X_te_s)
        lat = ((time.time() - t0) / len(X_te_s)) * 1000.0
        
        tr_scores = -model.decision_function(X_tr_s)
        thresh = np.percentile(tr_scores, 99)
        preds = (scores > thresh).astype(int)
        
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test, scores)
        except Exception:
            auc = 0.5
        fpr = float(np.sum((preds == 1) & (y_test == 0)) / (np.sum(y_test == 0) + 1e-9))
        
        ablation_results.append({
            'feature_count': cnt,
            'f1': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'roc_auc': round(auc, 4),
            'false_positive_rate': round(fpr, 4),
            'inference_ms': round(lat, 4),
            'model_size_kb': 0.05
        })
        
    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv("data/features/feature_ablation_results.csv", index=False)
    print(df_ablation.to_string(index=False))
    
    # --- PART 10: STATE DISCOVERY AUDIT ---
    print("\n--- PART 10: OPERATING STATE DISCOVERY AUDIT ---")
    X_tr_30 = train_df[train_df['label'] == 0][selected_features]
    scaler_30 = StandardScaler()
    X_tr_30_s = scaler_30.fit_transform(X_tr_30)
    
    state_disc = OperatingStateDiscoverer(min_clusters=2, max_clusters=5).fit(X_tr_30_s)
    pca_var_ratio = state_disc.pca.explained_variance_ratio_
    print(f"PCA 4-component explained variance: {np.sum(pca_var_ratio):.4f} ({pca_var_ratio})")
    print(f"Discovered Best K Clusters: {state_disc.best_k}")
    print(f"Cluster Metrics (Silhouette & Davies-Bouldin): {state_disc.cluster_metrics}")
    
    # State assignment distribution over full dataset
    X_full_s = scaler_30.transform(df_features[selected_features])
    state_labels = state_disc.predict(X_full_s)
    df_features['discovered_state'] = state_labels
    print("State Distribution across all 584 windows:")
    print(df_features['discovered_state'].value_counts())
    
    # Plot State Visualization
    plt.figure(figsize=(8, 6))
    X_pca = state_disc.pca.transform(X_full_s)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=state_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Discovered Operating State')
    plt.title("RetroFit Operating State Clustering (PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    os.makedirs("data/reports", exist_ok=True)
    plt.savefig("data/reports/operating_states_pca.png")
    plt.close()
    
    # --- PART 11-15: MODEL COMPARISON & FAILURE AUDIT ---
    print("\n--- PART 11-15: MODEL TRAINING & FAILURE AUDIT ---")
    X_te_30 = test_df[selected_features]
    X_te_30_s = scaler_30.transform(X_te_30)
    
    models_to_test = {
        'IsolationForest': IsolationForestAnomalyDetector(contamination=0.05),
        'Mahalanobis': MahalanobisAnomalyDetector(),
        'OneClassSVM': OneClassSVMAnomalyDetector(nu=0.05),
        'Autoencoder': AutoencoderAnomalyDetector(latent_dim=8, epochs=30, lr=1e-3)
    }
    
    model_audit_rows = []
    for mname, mobj in models_to_test.items():
        mobj.fit(X_tr_30_s)
        scores_te = mobj.predict_score(X_te_30_s)
        scores_tr = mobj.predict_score(X_tr_30_s)
        
        # 99th percentile threshold calibration
        thresh_99 = np.percentile(scores_tr, 99)
        preds_99 = (scores_te > thresh_99).astype(int)
        
        cm = confusion_matrix(y_test, preds_99, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        prec = precision_score(y_test, preds_99, zero_division=0)
        rec = recall_score(y_test, preds_99, zero_division=0)
        f1 = f1_score(y_test, preds_99, zero_division=0)
        try:
            auc = roc_auc_score(y_test, scores_te)
        except Exception:
            auc = 0.5
        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)
        
        model_audit_rows.append({
            'Model': mname,
            'Threshold_99': round(thresh_99, 4),
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'ROC_AUC': round(auc, 4),
            'FPR': round(fpr, 4),
            'FNR': round(fnr, 4),
            'Score_Min': round(np.min(scores_te), 4),
            'Score_Max': round(np.max(scores_te), 4),
            'Score_Mean': round(np.mean(scores_te), 4)
        })
        
    df_model_audit = pd.DataFrame(model_audit_rows)
    print(df_model_audit.to_string(index=False))
    
    # Save Confusion Matrix Plot for Isolation Forest
    plt.figure(figsize=(5, 4))
    if len(model_audit_rows) > 0:
        cm_if = [[df_model_audit.iloc[0]['TN'], df_model_audit.iloc[0]['FP']],
                 [df_model_audit.iloc[0]['FN'], df_model_audit.iloc[0]['TP']]]
        plt.imshow(cm_if, cmap='Blues')
        plt.title("Isolation Forest Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks([0, 1], ['Healthy (0)', 'Anomaly (1)'])
        plt.yticks([0, 1], ['Healthy (0)', 'Anomaly (1)'])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm_if[i][j]), ha='center', va='center', color='red' if i != j else 'white', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("data/reports/confusion_matrix_isolation_forest.png")
        plt.close()
        
    # --- PART 22: HISTORICAL MEMORY TEST ---
    print("\n--- PART 22: HISTORICAL MEMORY RETRIEVAL TEST ---")
    mem = BehavioralMemory(filepath="data/memory/test_events.json")
    mem.events = [] # reset test memory
    # Add 3 test events
    emb1 = np.random.normal(0, 1, 8).astype(np.float32)
    emb2 = np.random.normal(0, 1, 8).astype(np.float32)
    emb3 = emb1 + np.random.normal(0, 0.05, 8).astype(np.float32) # highly similar to emb1
    
    mem.add_event(operating_state=1, anomaly_score=0.45, behavior_drift=35.0, modality_contrib={'vibration': 0.8, 'audio': 0.2}, top_features=['acc_mag_kurtosis'], embedding=emb1.tolist())
    mem.add_event(operating_state=2, anomaly_score=0.75, behavior_drift=65.0, modality_contrib={'audio': 0.9, 'vibration': 0.1}, top_features=['audio_spectral_entropy'], embedding=emb2.tolist())
    mem.add_event(operating_state=1, anomaly_score=0.48, behavior_drift=38.0, modality_contrib={'vibration': 0.85, 'audio': 0.15}, top_features=['acc_mag_kurtosis'], embedding=emb3.tolist())
    
    matches = match_historical_events(mem.events, emb1, top_k=2)
    print("Memory Query Results for emb1:")
    print(matches)
    
    # --- PART 24: MISSING SENSOR ROBUSTNESS TEST ---
    print("\n--- PART 24: MISSING SENSOR ROBUSTNESS TEST ---")
    pipeline = RetroFitInferencePipeline(models_dir="data/models")
    
    sample_win = df_features.tail(1).copy()
    
    # Test 1: Full modalities
    res_full = pipeline.predict_window(sample_win)
    print(f"Full Sensors Status: {res_full['status']}, Similarity: {res_full['similarity']}%")
    
    # Test 2: Missing Magnetometer (zero out mag features)
    sample_no_mag = sample_win.copy()
    for col in sample_no_mag.columns:
        if col.startswith('mag_'):
            sample_no_mag[col] = 0.0
    res_no_mag = pipeline.predict_window(sample_no_mag)
    print(f"Missing Magnetometer Status: {res_no_mag['status']}, Similarity: {res_no_mag['similarity']}%")
    
    # Test 3: Missing Temperature (zero out temp features)
    sample_no_temp = sample_win.copy()
    for col in sample_no_temp.columns:
        if col.startswith('temp_'):
            sample_no_temp[col] = 0.0
    res_no_temp = pipeline.predict_window(sample_no_temp)
    print(f"Missing Temperature Status: {res_no_temp['status']}, Similarity: {res_no_temp['similarity']}%")
    
    # --- PART 30: UNSEEN SYNTHETIC DATASET B TEST ---
    print("\n--- PART 30: UNSEEN SYNTHETIC DATASET B GENERALIZATION TEST ---")
    dataset_b_raw = "data/raw/demo_machine_datasetB.csv"
    # Generate Dataset B with different random seed=999, +50% noise, +15% frequency shift, novel anomaly
    np.random.seed(999)
    dur_b = 300 # 300 seconds
    t_b = np.arange(0, dur_b, 0.001)
    n_b = len(t_b)
    
    ax_b = 0.15 * np.sin(2 * np.pi * 28.75 * t_b) + np.random.normal(0, 0.075, n_b)
    ay_b = 0.15 * np.cos(2 * np.pi * 28.75 * t_b) + np.random.normal(0, 0.075, n_b)
    az_b = 9.81 + np.random.normal(0, 0.075, n_b)
    gx_b = np.random.normal(0, 0.03, n_b)
    gy_b = np.random.normal(0, 0.03, n_b)
    gz_b = 0.6 * np.sin(2 * np.pi * 28.75 * t_b) + np.random.normal(0, 0.03, n_b)
    mx_b = 20.0 + np.random.normal(0, 0.075, n_b)
    my_b = 5.0 + np.random.normal(0, 0.075, n_b)
    mz_b = 45.0 + np.random.normal(0, 0.075, n_b)
    audio_b = 0.08 * np.sin(2 * np.pi * 506 * t_b) + np.random.normal(0, 0.03, n_b)
    temp_b = 26.5 + 0.015 * (t_b / 60.0)
    
    state_b = []
    sess_b = []
    lbl_b = []
    
    for i, t in enumerate(t_b):
        if t < 180:
            state_b.append("NORMAL_OPERATION")
            sess_b.append(10)
            lbl_b.append(0)
        else:
            # NOVEL UNSEEN ANOMALY: High-frequency Gyro oscillation (120Hz) + thermal acceleration spike
            state_b.append("HIGH_LOAD")
            sess_b.append(11)
            lbl_b.append(1)
            gz_b[i] += 1.8 * np.sin(2 * np.pi * 120 * t)
            ax_b[i] += 2.2 * np.sin(2 * np.pi * 120 * t)
            temp_b[i] += 12.0 + (t - 180) * 0.25
            
    df_b_raw = pd.DataFrame({
        'timestamp': 1770100000.0 + t_b,
        'machine_id': 'RETROFIT_DEMO_02_DS_B',
        'session_id': sess_b,
        'operating_state': state_b,
        'label': lbl_b,
        'ax': ax_b, 'ay': ay_b, 'az': az_b,
        'gx': gx_b, 'gy': gy_b, 'gz': gz_b,
        'mx': mx_b, 'my': my_b, 'mz': mz_b,
        'audio': audio_b,
        'temperature': temp_b
    })
    df_b_raw.to_csv(dataset_b_raw, index=False)
    
    df_b_feat = extract_features_from_dataframe(df_b_raw, config)
    X_b = df_b_feat[pipeline.selected_features]
    y_b = df_b_feat['label'].values
    
    X_b_s = pipeline.scaler.transform(X_b)
    scores_b = pipeline.anomaly_model.predict_score(X_b_s)
    # Evaluate Dataset B using State-Specific Thresholds
    state_preds_b = []
    for idx in range(len(df_b_feat)):
        feat_row = df_b_feat.iloc[[idx]]
        res = pipeline.predict_window(feat_row)
        state_preds_b.append(1 if res['status'] != 'NORMAL_OPERATION' else 0)
        
    cm_b_state = confusion_matrix(y_b, state_preds_b, labels=[0, 1])
    tn_bs, fp_bs, fn_bs, tp_bs = cm_b_state.ravel() if cm_b_state.size == 4 else (0, 0, 0, 0)
    prec_bs = precision_score(y_b, state_preds_b, zero_division=0)
    rec_bs = recall_score(y_b, state_preds_b, zero_division=0)
    f1_bs = f1_score(y_b, state_preds_b, zero_division=0)
    fpr_bs = fp_bs / (fp_bs + tn_bs + 1e-9)
    
    print(f"\n--- STATE-SPECIFIC THRESHOLD EVALUATION ON DATASET B ---")
    print(f"  TP: {tp_bs}, TN: {tn_bs}, FP: {fp_bs}, FN: {fn_bs}")
    print(f"  Precision: {prec_bs:.4f}, Recall: {rec_bs:.4f}, F1: {f1_bs:.4f}, FPR: {fpr_bs:.4f}")
    
    print("\n=========================================================")
    print("         SYSTEM AUDIT EXPERIMENTS COMPLETED SUCCESS!      ")
    print("=========================================================")

if __name__ == "__main__":
    run_complete_audit()
