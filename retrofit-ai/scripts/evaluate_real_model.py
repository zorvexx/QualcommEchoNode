import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import time
import pandas as pd
import numpy as np
from src.inference.inference import RetroFitInferencePipeline
from src.training.cross_session_pipeline import evaluate_predictions

def run_real_evaluation(features_csv="data/features/real_features.csv", models_dir="data/models"):
    print("=========================================================")
    print("  RETROFIT REAL LAPTOP INFERENCE & EVALUATION METRICS   ")
    print("=========================================================")
    
    pipeline = RetroFitInferencePipeline(models_dir=models_dir)
    print(f"[EVALUATE] Loaded Real Inference Pipeline from {models_dir}")
    print(f" -> Machine ID: {pipeline.fingerprint.get('machine_id', 'UNKNOWN')}")
    print(f" -> Selected Features ({len(pipeline.selected_features)}): {pipeline.selected_features[:5]}...")
    print(f" -> Global Threshold: {pipeline.threshold:.4f}")
    print(f" -> Behavioral Cluster Thresholds: {pipeline.state_thresholds}")
    
    df_features = pd.read_csv(features_csv)
    print(f"\n[EVALUATE] Running real inference over {len(df_features)} window feature vectors...")
    
    results = []
    latency_list = []
    
    for idx in range(len(df_features)):
        feat_win = df_features.iloc[[idx]]
        
        t0 = time.perf_counter()
        out = pipeline.predict_window(feat_win)
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000.0
        latency_list.append(latency_ms)
        
        sess_label = feat_win['session_label'].iloc[0] if 'session_label' in feat_win.columns else 'idle_01'
        true_label = feat_win['label'].iloc[0] if 'label' in feat_win.columns else 0
        
        results.append({
            'window_idx': idx,
            'session_label': sess_label,
            'behavioral_cluster': out['state'],
            'similarity': out['similarity'],
            'behavior_drift': out['behavior_drift'],
            'anomaly_score': out['anomaly_score'],
            'status': out['status'],
            'confidence': out['confidence'],
            'true_label': true_label,
            'pred_label': 1 if out['status'] != 'KNOWN_NORMAL_STATE' else 0,
            'latency_ms': round(latency_ms, 3)
        })
        
    df_res = pd.DataFrame(results)
    
    print("\n--- SAMPLE INFERENCE RESULTS (LAST 10 WINDOWS) ---")
    print(df_res[['window_idx', 'session_label', 'behavioral_cluster', 'similarity', 'anomaly_score', 'status', 'latency_ms']].tail(10).to_string(index=False))
    
    # Comprehensive Anomaly Detection Metrics
    y_true = df_res['true_label'].values
    y_pred = df_res['pred_label'].values
    y_scores = df_res['anomaly_score'].values
    
    metrics = evaluate_predictions(y_true, y_pred, y_scores)
    
    print("\n=========================================================")
    print("            REAL SYSTEM EVALUATION METRICS               ")
    print("=========================================================")
    print(f"  Total Windows Processed     : {len(df_res)}")
    print(f"  Average Inference Latency   : {np.mean(latency_list):.3f} ms / window")
    print(f"  95th Percentile Latency     : {np.percentile(latency_list, 95):.3f} ms / window")
    print(f"  Confusion Matrix (TN,FP,FN,TP): [{metrics['TN']}, {metrics['FP']}, {metrics['FN']}, {metrics['TP']}]")
    print(f"  Precision                   : {metrics['Precision']:.4f}")
    print(f"  Recall                      : {metrics['Recall']:.4f}")
    print(f"  F1-Score                    : {metrics['F1']:.4f}")
    print(f"  False Positive Rate (FPR)   : {metrics['FPR']:.4f}")
    print(f"  False Negative Rate (FNR)   : {metrics['FNR']:.4f}")
    print(f"  ROC-AUC                     : {metrics['ROC_AUC']:.4f}")
    print(f"  PR-AUC                      : {metrics['PR_AUC']:.4f}")
    
    # Status Taxonomy Breakdown
    print("\n--- THREE-TIER TAXONOMY BREAKDOWN ---")
    status_counts = df_res['status'].value_counts().to_dict()
    for s_name, s_cnt in status_counts.items():
        print(f"  {s_name:25s}: {s_cnt:4d} windows ({s_cnt/len(df_res)*100.1:.1f}%)")
        
    # Per-Session Breakdown
    print("\n--- PER-SESSION PERFORMANCE BREAKDOWN ---")
    for s_id, group in df_res.groupby('session_label'):
        normal_cnt = sum(group['status'] == 'KNOWN_NORMAL_STATE')
        unseen_cnt = sum(group['status'] == 'UNKNOWN_UNSEEN_BEHAVIOR')
        crit_cnt = sum(group['status'] == 'CRITICAL_ANOMALY')
        print(f"  Session '{s_id}': {len(group)} windows | Known Normal: {normal_cnt} | Unseen: {unseen_cnt} | Anomaly: {crit_cnt}")
        
    # Per-Behavioral-Cluster Breakdown
    print("\n--- PER-BEHAVIORAL-CLUSTER BREAKDOWN ---")
    for c_id, group in df_res.groupby('behavioral_cluster'):
        print(f"  Behavioral Cluster {c_id}: {len(group)} windows | Mean Score: {group['anomaly_score'].mean():.4f} | Std: {group['anomaly_score'].std():.4f}")
        
    print("\n=========================================================")
    print("        REAL EVALUATION & METRICS COMPLETED SUCCESS!     ")
    print("=========================================================")

if __name__ == "__main__":
    run_real_evaluation()
