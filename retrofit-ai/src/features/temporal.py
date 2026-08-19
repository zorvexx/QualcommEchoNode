import numpy as np

def compute_temporal_features(x, prefix=''):
    """
    Computes zero crossing rate, autocorrelation peak, lag correlation,
    jerk statistics, temporal variance, and temporal stability.
    """
    if len(x) == 0:
        return {}
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(x)) != 0)
    zcr = float(zero_crossings / (len(x) - 1))
    
    # Autocorrelation lag-1
    if len(x) > 1 and np.var(x) > 1e-9:
        autocorr_lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    else:
        autocorr_lag1 = 0.0
        
    # Jerk (derivative of signal)
    jerk = np.diff(x)
    jerk_rms = float(np.sqrt(np.mean(jerk**2))) if len(jerk) > 0 else 0.0
    jerk_std = float(np.std(jerk)) if len(jerk) > 0 else 0.0
    
    # Temporal stability / Variance
    temp_var = float(np.var(x))
    temp_stability = float(np.std(x) / (np.mean(np.abs(x)) + 1e-9))
    
    pfx = f"{prefix}_" if prefix else ""
    return {
        f"{pfx}zcr": zcr,
        f"{pfx}autocorr_lag1": autocorr_lag1,
        f"{pfx}jerk_rms": jerk_rms,
        f"{pfx}jerk_std": jerk_std,
        f"{pfx}temporal_variance": temp_var,
        f"{pfx}temporal_stability": temp_stability
    }
