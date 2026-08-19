import numpy as np

def calculate_behavioral_similarity(anomaly_score, threshold):
    """
    Calculates behavioral similarity % (0 - 100%) against baseline fingerprint threshold.
    Similarity = 100 * exp(-alpha * max(0, anomaly_score - threshold) / threshold)
    """
    if threshold <= 0:
        threshold = 1.0
    norm_score = anomaly_score / threshold
    if norm_score <= 1.0:
        return 100.0
    
    alpha = 0.693
    drift_factor = 1.0 - np.exp(-alpha * (norm_score - 1.0))
    similarity = (1.0 - drift_factor) * 100.0
    return max(0.0, min(100.0, float(similarity)))
