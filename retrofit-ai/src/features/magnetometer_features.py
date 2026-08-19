import numpy as np
from src.features.statistical import compute_statistical_features
from src.features.spectral import compute_spectral_features

def extract_magnetometer_features(mag_data, fs_mag=50):
    """
    Extracts optional magnetic magnitude, mean, variance, peak-to-peak, drift, spectral energy.
    mag_data: (N, 3) for Mx, My, Mz
    """
    features = {}
    if mag_data is None or len(mag_data) == 0:
        return features
    
    mx, my, mz = mag_data[:, 0], mag_data[:, 1], mag_data[:, 2]
    mmag = np.sqrt(mx**2 + my**2 + mz**2)
    
    features.update(compute_statistical_features(mmag, prefix='mag_mag'))
    features.update(compute_spectral_features(mmag, fs_mag, bands=[(0, 10), (10, 25)], prefix='mag_mag'))
    
    # Drift
    mag_drift = float(mmag[-1] - mmag[0])
    features['mag_drift'] = mag_drift
    
    return features
