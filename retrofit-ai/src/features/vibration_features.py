import numpy as np
from src.features.statistical import compute_statistical_features
from src.features.spectral import compute_spectral_features
from src.features.temporal import compute_temporal_features
from src.features.wavelet import compute_wavelet_features

def extract_vibration_features(accel_data, gyro_data, fs_accel=1000, fs_gyro=500, enable_wavelets=True):
    """
    Extracts statistical, spectral, temporal, and wavelet features for vibration (accel + gyro).
    accel_data: (N, 3) for Ax, Ay, Az
    gyro_data: (M, 3) for Gx, Gy, Gz
    """
    features = {}
    
    # 1. Accelerometer Combined Magnitude
    if accel_data is not None and len(accel_data) > 0:
        ax, ay, az = accel_data[:, 0], accel_data[:, 1], accel_data[:, 2]
        amag = np.sqrt(ax**2 + ay**2 + az**2)
        
        # Per-axis statistical features
        features.update(compute_statistical_features(ax, prefix='acc_x'))
        features.update(compute_statistical_features(ay, prefix='acc_y'))
        features.update(compute_statistical_features(az, prefix='acc_z'))
        features.update(compute_statistical_features(amag, prefix='acc_mag'))
        
        # Spectral features on amag
        vib_bands = [(0, 50), (50, 100), (100, 200), (200, 400)]
        features.update(compute_spectral_features(amag, fs_accel, bands=vib_bands, prefix='acc_mag'))
        
        # Temporal features on amag
        features.update(compute_temporal_features(amag, prefix='acc_mag'))
        
        # Wavelet features
        if enable_wavelets:
            features.update(compute_wavelet_features(amag, wavelet='db4', level=3, prefix='acc_mag'))
            
    # 2. Gyroscope Combined Magnitude
    if gyro_data is not None and len(gyro_data) > 0:
        gx, gy, gz = gyro_data[:, 0], gyro_data[:, 1], gyro_data[:, 2]
        gmag = np.sqrt(gx**2 + gy**2 + gz**2)
        features.update(compute_statistical_features(gmag, prefix='gyro_mag'))
        features.update(compute_spectral_features(gmag, fs_gyro, bands=[(0, 50), (50, 200)], prefix='gyro_mag'))
        
    return features
