"""
RetroFit Safe Adaptive Fingerprint Engine
Adapts the machine's healthy baseline to benign gradual drift (e.g. ambient thermal shifts,
mechanical burn-in) using Exponential Moving Average (EMA) while strictly freezing
the baseline during anomalies to avoid learning faults as normal.
"""

import numpy as np

class SafeAdaptiveFingerprint:
    def __init__(self, initial_fingerprint, learning_rate=0.01, max_drift_radius=2.5):
        """
        initial_fingerprint : 8-dim latent vector
        learning_rate       : EMA adaptation rate (e.g. 0.01 = 1% update per healthy window)
        max_drift_radius    : Maximum allowable distance from original factory baseline
        """
        self.factory_baseline = np.array(initial_fingerprint, dtype=float)
        self.current_fingerprint = np.array(initial_fingerprint, dtype=float)
        self.lr = learning_rate
        self.max_radius = max_drift_radius
        self.update_count = 0
        self.frozen_count = 0
        
    def step(self, window_latent_vector, is_anomaly=False):
        """
        Processes a new window latent embedding.
        If healthy: cautiously updates current_fingerprint via EMA.
        If anomaly: freezes update completely.
        """
        z = np.array(window_latent_vector, dtype=float)
        
        if is_anomaly:
            self.frozen_count += 1
            action = "FROZEN (Anomaly detected - baseline protected)"
        else:
            # Check distance from factory baseline
            candidate_fp = (1.0 - self.lr) * self.current_fingerprint + self.lr * z
            drift_from_factory = np.linalg.norm(candidate_fp - self.factory_baseline)
            
            if drift_from_factory <= self.max_radius:
                self.current_fingerprint = candidate_fp
                self.update_count += 1
                action = f"UPDATED (Safe drift {drift_from_factory:.3f}/{self.max_radius})"
            else:
                self.frozen_count += 1
                action = f"CAPPED (Max factory drift reached: {drift_from_factory:.3f})"
                
        return {
            'action': action,
            'current_fingerprint': self.current_fingerprint.tolist(),
            'drift_from_factory': float(np.linalg.norm(self.current_fingerprint - self.factory_baseline)),
            'update_count': self.update_count,
            'frozen_count': self.frozen_count
        }
