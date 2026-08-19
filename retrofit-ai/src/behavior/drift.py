import numpy as np

class BehaviorDriftTracker:
    """
    Calculates instantaneous and exponentially smoothed behavioral drift over time.
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.smoothed_drift = 0.0
        self.drift_history = []

    def update(self, similarity):
        instantaneous_drift = 100.0 - similarity
        if len(self.drift_history) == 0:
            self.smoothed_drift = instantaneous_drift
        else:
            self.smoothed_drift = self.alpha * instantaneous_drift + (1 - self.alpha) * self.smoothed_drift
            
        self.drift_history.append(self.smoothed_drift)
        return instantaneous_drift, self.smoothed_drift
