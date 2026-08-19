import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc
)

from src.selection.selector import select_top_features
from src.states.state_discovery import OperatingStateDiscoverer
from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.behavior.fingerprint import MachineFingerprint
from src.training.export_model import export_model_artifacts

def evaluate_predictions(y_true, y_pred, y_scores):
    """
    Computes comprehensive anomaly detection metrics suite.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    fpr = fp / (fp + tn + 1e-9)
    fnr = fn / (fn + tp + 1e-9)
    
    # ROC-AUC & PR-AUC
    try:
        if len(np.unique(y_true)) > 1:
            roc = roc_auc_score(y_true, y_scores)
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(r_curve, p_curve)
        else:
            roc = 0.5
            pr_auc = 0.5
    except Exception:
        roc = 0.5
        pr_auc = 0.5
        
    return {
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'Precision': round(float(prec), 4),
        'Recall': round(float(rec), 4),
        'F1': round(float(f1), 4),
        'FPR': round(float(fpr), 4),
        'FNR': round(float(fnr), 4),
        'ROC_AUC': round(float(roc), 4),
        'PR_AUC': round(float(pr_auc), 4)
    }

def run_cross_session_pipeline(df_features, train_sessions=None, test_sessions=None, top_n=30, output_dir="data/models"):
    """
    Executes a strict, non-leaking ML training & cross-session evaluation pipeline.
    """
    meta_cols = ['timestamp', 'machine_id', 'session_label', 'operating_state', 'label', 'eff_fs_win']
    feature_cols = [c for c in df_features.columns if c not in meta_cols]
    
    # Check session labels
    if 'session_label' not in df_features.columns:
        df_features['session_label'] = 'idle_01'
        
    # Split datasets without leakage
    if train_sessions is not None and test_sessions is not None:
        train_df = df_features[df_features['session_label'].isin(train_sessions)].copy()
        test_df = df_features[df_features['session_label'].isin(test_sessions)].copy()
        val_df = train_df.sample(frac=0.2, random_state=42) if len(train_df) >= 10 else train_df.copy()
    else:
        # Default Temporal Split: Train 60%, Val 20%, Test 20%
        n_total = len(df_features)
        n_tr = int(n_total * 0.6)
        n_val = int(n_total * 0.2)
        
        train_df = df_features.iloc[:n_tr].copy()
        val_df = df_features.iloc[n_tr:n_tr+n_val].copy()
        test_df = df_features.iloc[n_tr+n_val:].copy()
        
    print(f"\n[PIPELINE SPLIT] Strict Non-Leaking Partitioning:")
    print(f" -> Train Windows: {len(train_df)} (Sessions: {train_df['session_label'].unique().tolist()})")
    print(f" -> Val Windows:   {len(val_df)}")
    print(f" -> Test Windows:  {len(test_df)} (Sessions: {test_df['session_label'].unique().tolist()}) [HELD OUT UNSEEN]")
    
    # 1. Feature Selection ONLY on Train
    X_tr_raw = train_df[feature_cols]
    selected_features, ranking_df = select_top_features(X_tr_raw, top_n=top_n)
    print(f" -> Selected Top {len(selected_features)} Features (Fitted ONLY on Train)")
    
    # Pre-Training Hardware Validation Gate
    from src.training.validation_gate import validate_hardware_feature_compatibility
    validate_hardware_feature_compatibility(selected_features, strict=True)
    
    # 2. Scaler Fitting ONLY on Train
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(train_df[selected_features])
    X_val_scaled = scaler.transform(val_df[selected_features]) if len(val_df) > 0 else X_tr_scaled
    X_te_scaled = scaler.transform(test_df[selected_features]) if len(test_df) > 0 else X_tr_scaled
    
    # 3. Behavioral Cluster Discovery ONLY on Train
    cluster_discoverer = OperatingStateDiscoverer(min_clusters=2, max_clusters=4).fit(X_tr_scaled)
    print(f" -> Discovered {cluster_discoverer.best_k} Behavioral Clusters on Training Set")
    
    # 4. Anomaly Model Training ONLY on Train
    model = IsolationForestAnomalyDetector(contamination=0.05).fit(X_tr_scaled)
    tr_scores = model.predict_score(X_tr_scaled)
    
    # 5. State-Specific Threshold Calibration ONLY on Train
    tr_clusters = cluster_discoverer.predict(X_tr_scaled)
    global_thresh = float(np.percentile(tr_scores, 99))
    state_thresholds = {}
    for k in range(cluster_discoverer.best_k):
        mask_k = (tr_clusters == k)
        if np.sum(mask_k) >= 3:
            scores_k = tr_scores[mask_k]
            state_thresholds[str(k)] = max(global_thresh, float(np.percentile(scores_k, 99)))
        else:
            state_thresholds[str(k)] = global_thresh
            
    print(f" -> Calibrated Thresholds on Train: Global={global_thresh:.4f}, Clusters={state_thresholds}")
    
    # 6. Behavioral Fingerprint Construction
    fp_builder = MachineFingerprint(machine_id="LAPTOP_IDLE_01")
    fp_data = fp_builder.build_fingerprint(
        X_healthy=train_df[selected_features],
        selected_features=selected_features,
        state_model=cluster_discoverer,
        anomaly_threshold=global_thresh,
        state_thresholds=state_thresholds
    )
    
    # 7. Evaluate on Held-Out Test Set (Unseen)
    test_metrics = {}
    if len(test_df) > 0:
        te_scores = model.predict_score(X_te_scaled)
        te_clusters = cluster_discoverer.predict(X_te_scaled)
        te_preds = []
        for idx in range(len(test_df)):
            k_id = str(te_clusters[idx])
            t_k = state_thresholds.get(k_id, global_thresh)
            te_preds.append(1 if te_scores[idx] > t_k else 0)
            
        y_test = test_df['label'].values if 'label' in test_df.columns else np.zeros(len(test_df))
        test_metrics = evaluate_predictions(y_test, np.array(te_preds), te_scores)
        
        print("\n--- HELD-OUT UNSEEN TEST EVALUATION ---")
        for m_name, m_val in test_metrics.items():
            print(f"  {m_name:12s}: {m_val}")
            
    # Export artifacts
    export_model_artifacts(model, scaler, selected_features, cluster_discoverer, fp_data, output_dir=output_dir)
    
    return {
        'model': model,
        'scaler': scaler,
        'selected_features': selected_features,
        'cluster_discoverer': cluster_discoverer,
        'fingerprint': fp_data,
        'test_metrics': test_metrics
    }
