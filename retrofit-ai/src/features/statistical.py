import numpy as np
from scipy import stats

def compute_statistical_features(x, prefix=''):
    """
    Computes 17 time-domain statistical features for a 1D signal vector.
    """
    if len(x) == 0:
        return {}
    
    mean_val = float(np.mean(x))
    std_val = float(np.std(x))
    var_val = float(np.var(x))
    rms_val = float(np.sqrt(np.mean(x**2)))
    min_val = float(np.min(x))
    max_val = float(np.max(x))
    peak_val = float(np.max(np.abs(x)))
    peak_to_peak = float(max_val - min_val)
    median_val = float(np.median(x))
    
    q75, q25 = np.percentile(x, [75, 25])
    iqr_val = float(q75 - q25)
    
    skew_val = float(stats.skew(x)) if std_val > 1e-9 else 0.0
    kurtosis_val = float(stats.kurtosis(x)) if std_val > 1e-9 else 0.0
    
    abs_mean = float(np.mean(np.abs(x)))
    square_mean_root = float(np.mean(np.sqrt(np.abs(x))))**2
    
    crest_factor = float(peak_val / (rms_val + 1e-9))
    shape_factor = float(rms_val / (abs_mean + 1e-9))
    impulse_factor = float(peak_val / (abs_mean + 1e-9))
    clearance_factor = float(peak_val / (square_mean_root + 1e-9))
    sma_val = float(np.sum(np.abs(x)) / len(x))
    
    pfx = f"{prefix}_" if prefix else ""
    return {
        f"{pfx}mean": mean_val,
        f"{pfx}rms": rms_val,
        f"{pfx}variance": var_val,
        f"{pfx}std": std_val,
        f"{pfx}min": min_val,
        f"{pfx}max": max_val,
        f"{pfx}peak": peak_val,
        f"{pfx}peak_to_peak": peak_to_peak,
        f"{pfx}median": median_val,
        f"{pfx}iqr": iqr_val,
        f"{pfx}skewness": skew_val,
        f"{pfx}kurtosis": kurtosis_val,
        f"{pfx}crest_factor": crest_factor,
        f"{pfx}shape_factor": shape_factor,
        f"{pfx}impulse_factor": impulse_factor,
        f"{pfx}clearance_factor": clearance_factor,
        f"{pfx}sma": sma_val
    }
