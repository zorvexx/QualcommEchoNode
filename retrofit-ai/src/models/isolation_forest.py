import numpy as np
from sklearn.ensemble import IsolationForest

class IsolationForestAnomalyDetector:
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        
    def fit(self, X):
        self.model.fit(X)
        return self
        
    def predict_score(self, X):
        # Isolation Forest decision_function returns lower values for anomalies.
        # We invert it so higher score = higher anomaly.
        scores = -self.model.decision_function(X)
        return scores
