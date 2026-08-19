import numpy as np

def compute_feature_contributions(feature_names, residuals_or_deviations, top_k=5):
    """
    Ranks features by error residual magnitude and outputs top_k explanations.
    """
    paired = list(zip(feature_names, residuals_or_deviations))
    paired_sorted = sorted(paired, key=lambda x: abs(x[1]), reverse=True)
    
    top_features = [fname for fname, err in paired_sorted[:top_k]]
    contributions = {fname: round(float(err), 4) for fname, err in paired_sorted[:top_k]}
    return top_features, contributions
