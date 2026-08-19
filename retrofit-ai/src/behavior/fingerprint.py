import json
import numpy as np

class MachineFingerprint:
    """
    Builds and saves/loads machine-specific behavioral fingerprint.
    """
    def __init__(self, machine_id="MACHINE_01"):
        self.machine_id = machine_id
        self.fingerprint_data = {}

    def build_fingerprint(self, X_healthy, selected_features, state_model=None, anomaly_threshold=1.0, state_thresholds=None):
        means = X_healthy[selected_features].mean().to_dict()
        stds = X_healthy[selected_features].std().to_dict()
        mins = X_healthy[selected_features].min().to_dict()
        maxs = X_healthy[selected_features].max().to_dict()
        
        self.fingerprint_data = {
            'machine_id': self.machine_id,
            'num_samples': len(X_healthy),
            'selected_features': selected_features,
            'anomaly_threshold': anomaly_threshold,
            'state_thresholds': state_thresholds if state_thresholds is not None else {},
            'states': int(state_model.best_k) if state_model else 1,
            'feature_means': means,
            'feature_stds': stds,
            'feature_mins': mins,
            'feature_maxs': maxs
        }
        return self.fingerprint_data

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.fingerprint_data, f, indent=2)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            self.fingerprint_data = json.load(f)
        self.machine_id = self.fingerprint_data.get('machine_id', 'UNKNOWN')
        return self.fingerprint_data
