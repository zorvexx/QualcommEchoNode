import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix

from scripts.extract_real_features import run_real_feature_extraction
from src.hardware.capabilities import HardwareCapabilityRegistry
from src.training.validation_gate import validate_hardware_feature_compatibility
from src.selection.selector import select_top_features
from src.states.state_discovery import OperatingStateDiscoverer
from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.models.autoencoder import AutoencoderAnomalyDetector
from src.behavior.similarity import calculate_behavioral_similarity
from src.behavior.drift import BehaviorDriftTracker
from src.inference.decision_engine import DecisionEngine
from src.explainability.modality_contribution import compute_modality_contributions
from src.explainability.feature_contribution import compute_feature_contributions

def run_controlled_behavioral_experiment(csv_path="data/raw/unified_dataset_14col.csv", config_path="config.yaml"):
    print("=========================================================")
    print("      RETROFIT CONTROLLED BEHAVIORAL DEVIATION EXPERIMENT ")
    print("=========================================================")
    
    # 1. Extract feature vectors
    features_csv = "data/features/exp_real_features.csv"
    df_features = run_real_feature_extraction(csv_path, config_path, features_csv)
    
    # Calculate time in seconds relative to recording start
    t0 = df_features['timestamp'].iloc[0]
    t_sec = (df_features['timestamp'].values - t0) / 1000.0
    df_features['t_sec'] = t_sec
    
    # 2. Partition dataset into 5 EXACT temporal regions
    mask_train   = (t_sec >= 0.0)   & (t_sec <= 30.0)
    mask_per_a   = (t_sec > 30.0)   & (t_sec <= 40.0)
    mask_un_norm = (t_sec > 40.0)   & (t_sec <= 90.0)
    mask_per_b   = (t_sec > 90.0)   & (t_sec <= 100.0)
    mask_rec     = (t_sec > 100.0)  & (t_sec <= 137.2)
    
    df_train   = df_features[mask_train].copy()
    df_per_a   = df_features[mask_per_a].copy()
    df_un_norm = df_features[mask_un_norm].copy()
    df_per_b   = df_features[mask_per_b].copy()
    df_rec     = df_features[mask_rec].copy()
    
    print(f"\n[LEAK-FREE PARTITIONING CONTRACT]")
    print(f" -> TRAIN BASELINE (0-30s)      : {len(df_train)} windows [FITTING ONLY ON THIS]")
    print(f" -> PERTURBATION A (30-40s)     : {len(df_per_a)} windows [HELD OUT UNSEEN]")
    print(f" -> UNSEEN NORMAL (40-90s)      : {len(df_un_norm)} windows [HELD OUT UNSEEN]")
    print(f" -> PERTURBATION B (90-100s)    : {len(df_per_b)} windows [HELD OUT UNSEEN]")
    print(f" -> RECOVERY (100-137.2s)       : {len(df_rec)} windows [HELD OUT UNSEEN]")
    
    meta_cols = ['timestamp', 'machine_id', 'session_label', 'operating_state', 'label', 'eff_fs_win', 't_sec']
    candidate_cols = [c for c in df_features.columns if c not in meta_cols]
    
    # 3. Apply Hardware Capability Registry Gate
    registry = HardwareCapabilityRegistry()
    valid_candidate_cols = [c for c in candidate_cols if registry.is_feature_supported(c)]
    print(f"\n[HARDWARE CAPABILITY GATE] Filtered {len(candidate_cols)} candidate features -> {len(valid_candidate_cols)} physically valid features.")
    
    # 4. Feature Selection ONLY on TRAIN BASELINE (0-30s)
    X_tr_raw = df_train[valid_candidate_cols]
    selected_features, _ = select_top_features(X_tr_raw, top_n=30)
    
    # Pre-training hardware validation check
    validate_hardware_feature_compatibility(selected_features, strict=True)
    print(f" -> Selected Top 30 Features (Fitted ONLY on 0-30s Train): {selected_features[:5]}... (total {len(selected_features)})")
    
    # 5. Scaler Fitting ONLY on TRAIN BASELINE (0-30s)
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(df_train[selected_features])
    X_all_scaled = scaler.transform(df_features[selected_features])
    
    # 6. Behavioral Cluster Discovery ONLY on TRAIN BASELINE
    cluster_discoverer = OperatingStateDiscoverer(min_clusters=2, max_clusters=4).fit(X_tr_scaled)
    print(f" -> Discovered {cluster_discoverer.best_k} Behavioral Clusters on Train Set")
    
    # 7. Model Training ONLY on TRAIN BASELINE
    # A) Primary Model: Isolation Forest
    iso_model = IsolationForestAnomalyDetector(contamination=0.05).fit(X_tr_scaled)
    tr_iso_scores = iso_model.predict_score(X_tr_scaled)
    
    # B) Secondary Model: PyTorch Autoencoder (for comparison)
    ae_model = AutoencoderAnomalyDetector(epochs=100, lr=0.001).fit(X_tr_scaled)
    tr_ae_scores = ae_model.predict_score(X_tr_scaled)
    
    # 8. State-Specific Threshold Calibration ONLY on TRAIN BASELINE
    tr_clusters = cluster_discoverer.predict(X_tr_scaled)
    global_iso_thresh = float(np.percentile(tr_iso_scores, 99))
    global_ae_thresh = float(np.percentile(tr_ae_scores, 99))
    
    state_iso_thresh = {}
    for k in range(cluster_discoverer.best_k):
        mask_k = (tr_clusters == k)
        if np.sum(mask_k) >= 3:
            state_iso_thresh[str(k)] = max(global_iso_thresh, float(np.percentile(tr_iso_scores[mask_k], 99)))
        else:
            state_iso_thresh[str(k)] = global_iso_thresh
            
    print(f" -> Calibrated Thresholds: Global IsoThresh={global_iso_thresh:.4f}, StateIsoThresh={state_iso_thresh}")
    
    # 9. Execute Inference across ALL Window Frames
    all_iso_scores = iso_model.predict_score(X_all_scaled)
    all_ae_scores = ae_model.predict_score(X_all_scaled)
    all_clusters = cluster_discoverer.predict(X_all_scaled)
    
    drift_tracker = BehaviorDriftTracker(alpha=0.2)
    decision_engine = DecisionEngine()
    
    results = []
    for idx in range(len(df_features)):
        t_w = t_sec[idx]
        score_iso = all_iso_scores[idx]
        score_ae = all_ae_scores[idx]
        c_id = all_clusters[idx]
        t_k = state_iso_thresh.get(str(c_id), global_iso_thresh)
        
        sim = calculate_behavioral_similarity(score_iso, t_k)
        inst_d, smooth_d = drift_tracker.update(sim)
        
        # Override taxonomy status logic
        if score_iso > 3.0 * t_k:
            status = "CRITICAL_ANOMALY"
        elif score_iso > t_k:
            status = "UNKNOWN_UNSEEN_BEHAVIOR"
        else:
            status = "KNOWN_NORMAL_STATE"
            
        # Residual & Modality attributions
        diff = (X_all_scaled[idx] - np.mean(X_tr_scaled, axis=0))**2
        mod_contrib = compute_modality_contributions(selected_features, diff)
        top_f, f_contrib = compute_feature_contributions(selected_features, diff, top_k=3)
        
        # Region label
        if 0.0 <= t_w <= 30.0:
            region = "TRAIN_BASELINE"
        elif 30.0 < t_w <= 40.0:
            region = "PERTURBATION_A"
        elif 40.0 < t_w <= 90.0:
            region = "UNSEEN_NORMAL"
        elif 90.0 < t_w <= 100.0:
            region = "PERTURBATION_B"
        else:
            region = "RECOVERY"
            
        results.append({
            'idx': idx,
            't_sec': round(t_w, 2),
            'region': region,
            'cluster': int(c_id),
            'iso_score': float(score_iso),
            'ae_score': float(score_ae),
            'threshold': float(t_k),
            'similarity': float(sim),
            'drift': float(smooth_d),
            'status': status,
            'vib_pct': float(mod_contrib.get('vibration', 0.0)),
            'therm_pct': float(mod_contrib.get('thermal', 0.0)),
            'aud_pct': float(mod_contrib.get('acoustic', 0.0)),
            'top_feature_1': top_f[0] if len(top_f) > 0 else '',
            'top_feature_2': top_f[1] if len(top_f) > 1 else '',
            'top_feature_3': top_f[2] if len(top_f) > 2 else ''
        })
        
    df_res = pd.DataFrame(results)
    
    # -------------------------------------------------------------
    # REGION EVALUATION SUMMARY
    # -------------------------------------------------------------
    regions = ['TRAIN_BASELINE', 'PERTURBATION_A', 'UNSEEN_NORMAL', 'PERTURBATION_B', 'RECOVERY']
    region_summary = {}
    
    print("\n=========================================================")
    print("         PER-REGION EXPERIMENTAL EVALUATION RESULTS       ")
    print("=========================================================")
    
    for reg in regions:
        sub = df_res[df_res['region'] == reg]
        n_win = len(sub)
        status_counts = sub['status'].value_counts().to_dict()
        n_norm = status_counts.get('KNOWN_NORMAL_STATE', 0)
        n_unseen = status_counts.get('UNKNOWN_UNSEEN_BEHAVIOR', 0)
        n_crit = status_counts.get('CRITICAL_ANOMALY', 0)
        
        iso_mean = float(sub['iso_score'].mean()) if n_win > 0 else 0.0
        iso_std  = float(sub['iso_score'].std()) if n_win > 0 else 0.0
        iso_min  = float(sub['iso_score'].min()) if n_win > 0 else 0.0
        iso_max  = float(sub['iso_score'].max()) if n_win > 0 else 0.0
        
        top_v = sub['vib_pct'].mean()
        top_t = sub['therm_pct'].mean()
        top_a = sub['aud_pct'].mean()
        
        region_summary[reg] = {
            'n_windows': n_win,
            'decisions': {'KNOWN_NORMAL_STATE': n_norm, 'UNKNOWN_UNSEEN_BEHAVIOR': n_unseen, 'CRITICAL_ANOMALY': n_crit},
            'iso_score_dist': {'mean': round(iso_mean, 6), 'std': round(iso_std, 6), 'min': round(iso_min, 6), 'max': round(iso_max, 6)},
            'modality_avg': {'vibration_pct': round(top_v, 1), 'thermal_pct': round(top_t, 1), 'acoustic_pct': round(top_a, 1)}
        }
        
        print(f"\n--- Region: {reg} ({n_win} Windows) ---")
        print(f"  Decisions        : Normal={n_norm} | UnseenBehavior={n_unseen} | CriticalAnomaly={n_crit}")
        print(f"  IsoScore Dist    : Mean={iso_mean:.6f} (std {iso_std:.6f}) [Min: {iso_min:.6f}, Max: {iso_max:.6f}]")
        print(f"  Avg Modality     : Vibration={top_v:.1f}% | Thermal={top_t:.1f}% | Acoustic={top_a:.1f}%")

    # Metrics on Unseen Normal & Perturbations
    unseen_norm_sub = df_res[df_res['region'] == 'UNSEEN_NORMAL']
    fp_count = len(unseen_norm_sub[unseen_norm_sub['status'] != 'KNOWN_NORMAL_STATE'])
    fpr = fp_count / (len(unseen_norm_sub) + 1e-9)
    
    per_a_sub = df_res[df_res['region'] == 'PERTURBATION_A']
    det_a_count = len(per_a_sub[per_a_sub['status'] != 'KNOWN_NORMAL_STATE'])
    det_rate_a = det_a_count / (len(per_a_sub) + 1e-9)
    
    per_b_sub = df_res[df_res['region'] == 'PERTURBATION_B']
    det_b_count = len(per_b_sub[per_b_sub['status'] != 'KNOWN_NORMAL_STATE'])
    det_rate_b = det_b_count / (len(per_b_sub) + 1e-9)
    
    print("\n--- KEY EXPERIMENTAL METRICS ---")
    print(f"  False Positive Rate on Unseen Normal (40-90s) : {fpr*100:.2f}% ({fp_count}/{len(unseen_norm_sub)} windows)")
    print(f"  Behavioral Deviation Detection Rate Perturbation A (30-40s) : {det_rate_a*100:.2f}% ({det_a_count}/{len(per_a_sub)} windows)")
    print(f"  Behavioral Deviation Detection Rate Perturbation B (90-100s): {det_rate_b*100:.2f}% ({det_b_count}/{len(per_b_sub)} windows)")

    # -------------------------------------------------------------
    # GENERATE 5 EXPERIMENTAL PLOTS
    # -------------------------------------------------------------
    plt.style.use('dark_background')
    plot_dir = "data/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Confusion Matrix Plot (Normal vs Perturbation Deviation)
    y_true_exp = []
    y_pred_exp = []
    for idx, row in df_res.iterrows():
        reg = row['region']
        if reg in ['TRAIN_BASELINE', 'UNSEEN_NORMAL', 'RECOVERY']:
            y_true_exp.append(0) # Expected Baseline/Normal
        else:
            y_true_exp.append(1) # Expected Physical Perturbation
            
        y_pred_exp.append(0 if row['status'] == 'KNOWN_NORMAL_STATE' else 1)
        
    cm = confusion_matrix(y_true_exp, y_pred_exp, labels=[0, 1])
    
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
    plt.xticks([0, 1], ['Normal', 'Behavioral Shift'])
    plt.yticks([0, 1], ['Normal', 'Perturbation'])
    plt.title("RetroFit Perturbation Experiment — Confusion Matrix")
    plt.xlabel("Predicted Status")
    plt.ylabel("Actual Temporal Region")
    p1 = os.path.join(plot_dir, "exp_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()
    
    # 2. Score vs Time Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df_res['t_sec'], df_res['iso_score'], label='Isolation Forest Score', color='#38bdf8', linewidth=1.5)
    plt.plot(df_res['t_sec'], df_res['ae_score'], label='Autoencoder Score (Comparison)', color='#a78bfa', linestyle='--', linewidth=1.2)
    plt.axhline(global_iso_thresh, color='#f43f5e', linestyle=':', label=f'Global Threshold ({global_iso_thresh:.4f})')
    
    # Shade Regions
    plt.axvspan(0, 30, color='#22c55e', alpha=0.1, label='Train Baseline (0-30s)')
    plt.axvspan(30, 40, color='#fbbf24', alpha=0.2, label='Perturbation A (30-40s)')
    plt.axvspan(40, 90, color='#38bdf8', alpha=0.1, label='Unseen Normal (40-90s)')
    plt.axvspan(90, 100, color='#f43f5e', alpha=0.2, label='Perturbation B (90-100s)')
    plt.axvspan(100, 137.2, color='#94a3b8', alpha=0.1, label='Recovery (100-137s)')
    
    plt.title("RetroFit Anomaly Score vs Time Across Temporal Regions")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Anomaly Score")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', fontsize=8)
    p2 = os.path.join(plot_dir, "exp_score_vs_time.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()

    # 3. Status vs Time Plot
    plt.figure(figsize=(12, 4))
    status_num_map = {'KNOWN_NORMAL_STATE': 0, 'UNKNOWN_UNSEEN_BEHAVIOR': 1, 'CRITICAL_ANOMALY': 2}
    df_res['status_code'] = df_res['status'].map(status_num_map)
    
    plt.step(df_res['t_sec'], df_res['status_code'], where='post', color='#f43f5e', linewidth=2.0)
    plt.yticks([0, 1, 2], ['KNOWN_NORMAL', 'UNSEEN_BEHAVIOR', 'CRITICAL_ANOMALY'])
    plt.title("RetroFit Three-Tier Status Taxonomy vs Time")
    plt.xlabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.3)
    p3 = os.path.join(plot_dir, "exp_status_vs_time.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()

    # 4. Modality Contribution vs Time Plot
    plt.figure(figsize=(12, 5))
    plt.stackplot(df_res['t_sec'], df_res['vib_pct'], df_res['therm_pct'], df_res['aud_pct'],
                  labels=['Vibration %', 'Thermal %', 'Acoustic %'],
                  colors=['#4ade80', '#f43f5e', '#a78bfa'], alpha=0.8)
    plt.title("RetroFit Modality Contribution Breakdown vs Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Modality Contribution (%)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    p4 = os.path.join(plot_dir, "exp_modality_vs_time.png")
    plt.tight_layout()
    plt.savefig(p4, dpi=150)
    plt.close()

    # 5. Top Feature Attribution Plot
    top_feature_counts = pd.Series(df_res['top_feature_1'].tolist() + df_res['top_feature_2'].tolist()).value_counts().head(10)
    plt.figure(figsize=(10, 5))
    top_feature_counts.plot(kind='barh', color='#38bdf8')
    plt.title("RetroFit Top Contributing Feature Frequency Across Experiment")
    plt.xlabel("Count of Occurrences as Top-3 Contributor")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.3)
    p5 = os.path.join(plot_dir, "exp_top_features.png")
    plt.tight_layout()
    plt.savefig(p5, dpi=150)
    plt.close()

    print("\n--- GENERATED EXPERIMENTAL PLOTS ---")
    print(f"  1. Confusion Matrix       -> {p1}")
    print(f"  2. Score vs Time          -> {p2}")
    print(f"  3. Status vs Time         -> {p3}")
    print(f"  4. Modality vs Time       -> {p4}")
    print(f"  5. Top Feature Attribution-> {p5}")

    # Export experiment metrics JSON
    exp_json = "data/reports/controlled_experiment_report.json"
    os.makedirs(os.path.dirname(exp_json), exist_ok=True)
    with open(exp_json, 'w') as f:
        json.dump({
            'selected_features': selected_features,
            'fpr_unseen_normal': round(fpr, 4),
            'det_rate_pert_a': round(det_rate_a, 4),
            'det_rate_pert_b': round(det_rate_b, 4),
            'region_summary': region_summary
        }, f, indent=2)
        
    print(f"\n[EXPERIMENT] Saved complete report to {exp_json}")
    return df_res

if __name__ == "__main__":
    run_controlled_behavioral_experiment()
