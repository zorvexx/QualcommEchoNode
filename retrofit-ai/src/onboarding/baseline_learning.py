import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.hardware.capabilities import HardwareCapabilityRegistry
from src.training.validation_gate import validate_hardware_feature_compatibility
from src.selection.selector import select_top_features
from src.states.state_discovery import OperatingStateDiscoverer
from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.behavior.fingerprint import MachineFingerprint
from src.registration.manager import MachineRegistrationManager
from src.training.export_model import export_model_artifacts

class AdaptiveBaselineOnboarder:
    """
    Implements Adaptive Machine Onboarding & Baseline Learning mechanism.
    Enforces minimum statistical sampling guards, timestamp quality checks,
    baseline stability validation, disturbance rejection, and thermal warmup handling.
    """
    def __init__(self,
                 min_duration_sec=120.0,
                 min_windows=50,
                 max_gap_ms=200.0,
                 min_fs_hz=50.0,
                 max_contamination_pct=15.0):
        self.min_duration_sec = min_duration_sec
        self.min_windows = min_windows
        self.max_gap_ms = max_gap_ms
        self.min_fs_hz = min_fs_hz
        self.max_contamination_pct = max_contamination_pct
        self.registry = HardwareCapabilityRegistry()

    def check_baseline_sufficiency(self, df_features, timestamps_ms):
        """
        Validates minimum duration and sample window count guard.
        Raises ValueError if baseline is insufficient for robust feature selection.
        """
        if len(timestamps_ms) < 2:
            raise ValueError("INSUFFICIENT BASELINE DATA: At least 2 timestamped samples required.")
            
        dur_sec = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
        n_windows = len(df_features)
        
        print(f"[ONBOARDING CHECK] Duration: {dur_sec:.1f}s / Min: {self.min_duration_sec:.1f}s | Windows: {n_windows} / Min: {self.min_windows}")
        
        if dur_sec < self.min_duration_sec or n_windows < self.min_windows:
            raise ValueError(
                f"INSUFFICIENT BASELINE DATA: Baseline requires min {self.min_duration_sec:.1f}s duration "
                f"and min {self.min_windows} windows. Current recording has {dur_sec:.1f}s and {n_windows} windows. "
                f"Model training aborted to prevent unstable feature selection and high false alarms."
            )
            
        return dur_sec, n_windows

    def analyze_baseline_stability(self, df_features, df_raw=None):
        """
        Evaluates baseline stability, timestamp regularity, contamination ratio,
        and thermal warmup trajectory.
        """
        stability_issues = []
        contamination_flags = np.zeros(len(df_features), dtype=bool)
        
        # 1. Timestamp & Sampling Quality
        if 'timestamp' in df_features.columns:
            ts = df_features['timestamp'].values
            dt = np.diff(ts)
            max_dt = np.max(dt) if len(dt) > 0 else 0
            if max_dt > self.max_gap_ms:
                stability_issues.append(f"Excessive timestamp gap detected: {max_dt:.1f}ms > {self.max_gap_ms}ms")
                
        # 2. Contamination / Disturbance Rejection
        # Check vibration variance spikes relative to baseline median
        if 'acc_mag_std' in df_features.columns:
            stds = df_features['acc_mag_std'].values
            lower_subset = stds[stds <= np.percentile(stds, 75)]
            med_std = np.median(lower_subset) if len(lower_subset) > 0 else np.median(stds)
            std_std = np.std(lower_subset) if len(lower_subset) > 0 else np.std(stds)
            thresh_std = med_std + 4.0 * (std_std + 10.0)
            contamination_flags = stds > thresh_std
            
        n_contaminated = int(np.sum(contamination_flags))
        contam_pct = (n_contaminated / len(df_features)) * 100.0
        
        if contam_pct > self.max_contamination_pct:
            stability_issues.append(f"Baseline contamination high: {contam_pct:.1f}% windows disturbed (> {self.max_contamination_pct}%)")

        # 3. Thermal Warmup Slope Analysis
        thermal_slope = 0.0
        if 'temp_object_mean' in df_features.columns or 'temp_ambient_mean' in df_features.columns:
            t_col = 'temp_object_mean' if 'temp_object_mean' in df_features.columns else 'temp_ambient_mean'
            temps = df_features[t_col].values
            if len(temps) > 1:
                t_duration = (df_features['timestamp'].iloc[-1] - df_features['timestamp'].iloc[0]) / 1000.0 if 'timestamp' in df_features.columns else len(df_features)
                thermal_slope = (temps[-1] - temps[0]) / (t_duration + 1e-6) # °C per second
                print(f" -> Thermal Warmup Trajectory: Slope = {thermal_slope:.6f} °C/s (Total Rise: {temps[-1] - temps[0]:.2f}°C)")

        # Overall Baseline Stability Score (0.0 to 100.0%)
        stability_score = max(0.0, 100.0 - contam_pct * 2.0 - (10.0 if len(stability_issues) > 0 else 0.0))
        
        return {
            'is_stable': contam_pct <= self.max_contamination_pct and len(stability_issues) == 0,
            'stability_score': round(float(stability_score), 1),
            'contamination_pct': round(float(contam_pct), 1),
            'contaminated_flags': contamination_flags,
            'thermal_slope_c_per_sec': round(float(thermal_slope), 6),
            'issues': stability_issues
        }

    def onboard_machine_baseline(self,
                                 df_features,
                                 machine_id,
                                 machine_type="General Workstation",
                                 top_n=30,
                                 output_dir="data/machines"):
        """
        Executes adaptive machine onboarding workflow.
        Returns onboarding metadata dict and saves trained model artifacts.
        """
        reg_mgr = MachineRegistrationManager(base_dir=output_dir)
        reg_mgr.update_lifecycle_state(machine_id, "LEARNING")
        
        ts_col = 'timestamp' if 'timestamp' in df_features.columns else df_features.columns[0]
        timestamps_ms = df_features[ts_col].values.astype(float)
        
        # Step 1: Data Sufficiency Guard Check
        try:
            dur_sec, n_windows = self.check_baseline_sufficiency(df_features, timestamps_ms)
        except ValueError as e:
            reg_mgr.update_lifecycle_state(machine_id, "BASELINE_NOT_READY")
            raise e
        
        # Step 2: Baseline Stability Analysis & Contamination Filtering
        stability_res = self.analyze_baseline_stability(df_features)
        
        if not stability_res['is_stable']:
            reg_mgr.update_lifecycle_state(machine_id, "BASELINE_NOT_READY")
            print(f"[ONBOARDING REJECTED] Machine '{machine_id}' baseline is disturbed/unstable. Issues: {stability_res['issues']}")
            raise ValueError(f"BASELINE NOT READY for Machine '{machine_id}': {', '.join(stability_res['issues'])}")

        # Filter out contaminated windows from baseline training set
        clean_mask = ~stability_res['contaminated_flags']
        df_clean_baseline = df_features[clean_mask].copy()
        
        print(f"[BASELINE CLEANED] Retained {len(df_clean_baseline)} / {n_windows} clean baseline windows for training.")

        # Step 3: Hardware Capability Gate & Feature Selection ONLY on Clean Baseline
        meta_cols = ['timestamp', 'machine_id', 'session_label', 'operating_state', 'label', 'eff_fs_win', 't_sec']
        candidate_cols = [c for c in df_clean_baseline.columns if c not in meta_cols]
        
        valid_candidate_cols = [c for c in candidate_cols if self.registry.is_feature_supported(c)]
        print(f" -> Hardware Capability Gate: Filtered {len(candidate_cols)} candidate features -> {len(valid_candidate_cols)} physically valid features.")
        
        X_tr_raw = df_clean_baseline[valid_candidate_cols]
        selected_features, _ = select_top_features(X_tr_raw, top_n=top_n)
        
        # Enforce pre-training hardware gate
        validate_hardware_feature_compatibility(selected_features, strict=True)
        
        # Step 4: Scaler, Multi-State Cluster Discovery (Thermal Aware), & Isolation Forest Model Fitting
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(df_clean_baseline[selected_features])
        
        # Dynamic Cluster Discovery (learns 2-4 operating clusters including thermal warmup states)
        cluster_discoverer = OperatingStateDiscoverer(min_clusters=2, max_clusters=4).fit(X_tr_scaled)
        print(f" -> Discovered {cluster_discoverer.best_k} Behavioral/Thermal Operating Clusters on Baseline")
        
        iso_model = IsolationForestAnomalyDetector(contamination=0.05).fit(X_tr_scaled)
        tr_scores = iso_model.predict_score(X_tr_scaled)
        
        # Calibrate state-specific thresholds
        tr_clusters = cluster_discoverer.predict(X_tr_scaled)
        global_thresh = float(np.percentile(tr_scores, 99))
        state_thresholds = {}
        for k in range(cluster_discoverer.best_k):
            mask_k = (tr_clusters == k)
            if np.sum(mask_k) >= 3:
                state_thresholds[str(k)] = max(global_thresh, float(np.percentile(tr_scores[mask_k], 99)))
            else:
                state_thresholds[str(k)] = global_thresh
                
        # Step 5: Machine Fingerprint Construction
        fp_builder = MachineFingerprint(machine_id=machine_id)
        fp_data = fp_builder.build_fingerprint(
            X_healthy=df_clean_baseline[selected_features],
            selected_features=selected_features,
            state_model=cluster_discoverer,
            anomaly_threshold=global_thresh,
            state_thresholds=state_thresholds
        )
        
        # Step 6: Export Machine-Specific Model Artifacts & Onboarding Metadata
        machine_model_dir = os.path.join(output_dir, machine_id)
        os.makedirs(machine_model_dir, exist_ok=True)
        
        export_model_artifacts(
            model=iso_model,
            scaler=scaler,
            selected_features=selected_features,
            state_model=cluster_discoverer,
            fingerprint_data=fp_data,
            output_dir=machine_model_dir
        )
        
        onboarding_metadata = {
            'machine_id': machine_id,
            'machine_type': machine_type,
            'baseline_duration_sec': round(dur_sec, 2),
            'num_windows': n_windows,
            'clean_windows_used': len(df_clean_baseline),
            'baseline_stability_score': stability_res['stability_score'],
            'thermal_warmup_slope_c_per_sec': stability_res['thermal_slope_c_per_sec'],
            'selected_features': selected_features,
            'discovered_behavioral_clusters': cluster_discoverer.best_k,
            'fingerprint_version': 'v2.0-ADAPTIVE_ONBOARDED',
            'onboarding_status': 'BASELINE_READY',
            'lifecycle_state': 'MONITORING',
            'state_thresholds': state_thresholds,
            'global_threshold': global_thresh
        }
        
        meta_path = os.path.join(machine_model_dir, "onboarding_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(onboarding_metadata, f, indent=2)
            
        reg_mgr.update_lifecycle_state(machine_id, "BASELINE_READY")
        reg_mgr.update_lifecycle_state(machine_id, "MONITORING")
        
        print(f"[ONBOARDING SUCCESS] Machine '{machine_id}' successfully onboarded into MONITORING state.")
        return onboarding_metadata
