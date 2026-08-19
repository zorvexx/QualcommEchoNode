import time
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.models.mahalanobis import MahalanobisAnomalyDetector
from src.models.one_class_svm import OneClassSVMAnomalyDetector
from src.models.autoencoder import AutoencoderAnomalyDetector
from src.training.cross_session_pipeline import evaluate_predictions

def benchmark_models_on_real_data(df_features, top_n=30):
    """
    Benchmarks anomaly candidates on real held-out session data using strict leak-free split.
    Keeps synthetic results clearly separate from real data results.
    """
    meta_cols = ['timestamp', 'machine_id', 'session_label', 'operating_state', 'label', 'eff_fs_win']
    feature_cols = [c for c in df_features.columns if c not in meta_cols]
    
    # Filter valid non-constant features
    var_mask = (df_features[feature_cols].std() > 1e-6)
    valid_features = list(df_features[feature_cols].columns[var_mask])
    
    # Strict Leak-Free Temporal Split: Train 60%, Val 20%, Test 20%
    n_total = len(df_features)
    n_tr = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    
    train_df = df_features.iloc[:n_tr].copy()
    val_df = df_features.iloc[n_tr:n_tr+n_val].copy()
    test_df = df_features.iloc[n_tr+n_val:].copy()
    
    # 1. Feature Selection ONLY on Train
    X_tr_raw = train_df[valid_features]
    var_series = X_tr_raw.var() / (X_tr_raw.abs().mean() + 1e-9)
    selected_features = var_series.nlargest(min(top_n, len(valid_features))).index.tolist()
    
    # 2. Scaler Fitting ONLY on Train
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(train_df[selected_features])
    X_val_s = scaler.transform(val_df[selected_features]) if len(val_df) > 0 else X_tr_s
    X_te_s = scaler.transform(test_df[selected_features]) if len(test_df) > 0 else X_tr_s
    
    y_test = test_df['label'].values if 'label' in test_df.columns else np.zeros(len(test_df))
    
    candidates = {
        'IsolationForest': IsolationForestAnomalyDetector(contamination=0.05),
        'RobustMahalanobis': MahalanobisAnomalyDetector(),
        'OneClassSVM': OneClassSVMAnomalyDetector(nu=0.05),
        'Autoencoder': AutoencoderAnomalyDetector(latent_dim=4, epochs=20, lr=1e-3)
    }
    
    results = []
    
    for name, model_obj in candidates.items():
        # Fit ONLY on Train
        t_start = time.perf_counter()
        model_obj.fit(X_tr_s)
        t_fit = (time.perf_counter() - t_start) * 1000.0
        
        # Calculate threshold on Train (99th percentile)
        tr_scores = model_obj.predict_score(X_tr_s)
        thresh_99 = float(np.percentile(tr_scores, 99))
        
        # Inference Latency on Test
        latencies = []
        te_scores = []
        for i in range(len(X_te_s)):
            row = X_te_s[[i]]
            t0 = time.perf_counter()
            s = model_obj.predict_score(row)[0]
            latencies.append((time.perf_counter() - t0) * 1000.0)
            te_scores.append(s)
            
        te_scores = np.array(te_scores)
        te_preds = (te_scores > thresh_99).astype(int)
        
        metrics = evaluate_predictions(y_test, te_preds, te_scores)
        
        avg_lat = np.mean(latencies)
        
        # Estimate Model Size (KB)
        tmp_file = f"data/models/tmp_{name}.pkl"
        os.makedirs(os.path.dirname(tmp_file), exist_ok=True)
        joblib.dump(model_obj, tmp_file)
        size_kb = os.path.getsize(tmp_file) / 1024.0
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
            
        # Uno Q Feasibility
        uno_q_feasible = "YES (Highly Recommended)" if size_kb < 100 and avg_lat < 1.0 else ("FEASIBLE" if size_kb < 500 else "HEAVY")
        
        results.append({
            'Model': name,
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1_Score': metrics['F1'],
            'FPR': metrics['FPR'],
            'FNR': metrics['FNR'],
            'PR_AUC': metrics['PR_AUC'],
            'ROC_AUC': metrics['ROC_AUC'],
            'Inference_ms': round(avg_lat, 3),
            'Model_Size_KB': round(size_kb, 2),
            'Uno_Q_Feasibility': uno_q_feasible
        })
        
    df_results = pd.DataFrame(results)
    return df_results
