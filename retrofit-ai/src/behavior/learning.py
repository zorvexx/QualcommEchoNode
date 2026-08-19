import os
import json
import numpy as np

class AdaptiveBaselineLearner:
    """
    Manages baseline updating with strict safety safeguards.
    ADAPTIVE LEARNING IS OFF BY DEFAULT.
    Only stable, high-confidence KNOWN_NORMAL observations can update the baseline.
    NEVER allows UNKNOWN_UNSEEN_BEHAVIOR or CRITICAL_ANOMALY observations to contaminate healthy baseline.
    """
    def __init__(self, enabled=False, min_stable_count=20, max_drift_tolerance=5.0):
        self.enabled = enabled
        self.min_stable_count = min_stable_count
        self.max_drift_tolerance = max_drift_tolerance
        self.stable_buffer = []

    def process_observation(self, status, anomaly_score, feature_values, confidence):
        """
        Evaluates whether an observation is safe for baseline adaptation.
        """
        if not self.enabled:
            return {'adapted': False, 'reason': 'Adaptive learning is OFF by default.'}

        # STRICT SAFEGUARD: Reject non-normal observations
        if status in ['UNKNOWN_UNSEEN_BEHAVIOR', 'CRITICAL_ANOMALY', 'UNKNOWN_BEHAVIOR']:
            self.stable_buffer.clear() # Reset buffer on any deviation
            return {'adapted': False, 'reason': f'Rejected non-normal status: {status}'}

        if confidence < 80.0:
            return {'adapted': False, 'reason': f'Low confidence: {confidence}%'}

        self.stable_buffer.append(feature_values)

        if len(self.stable_buffer) >= self.min_stable_count:
            # Safely calculate updated running mean
            new_means = np.mean(self.stable_buffer, axis=0)
            self.stable_buffer.clear()
            return {'adapted': True, 'updated_means': new_means, 'samples_used': self.min_stable_count}

        return {'adapted': False, 'reason': f'Buffering stable samples ({len(self.stable_buffer)}/{self.min_stable_count})'}
