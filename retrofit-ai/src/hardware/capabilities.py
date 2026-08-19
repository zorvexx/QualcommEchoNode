import os
import json

class HardwareCapabilityRegistry:
    """
    Registry defining physically valid features supported by the solder sensor node:
    - GY-87 / MPU6050 (ax, ay, az, gx, gy, gz)
    - MAX9814 (amplitude-only: sound_peak, sound_volts)
    - MLX90614 (ir_ambient_c, ir_object_c)
    - Magnetometer (mx, my, mz - PERMANENTLY DISABLED)
    """
    def __init__(self):
        self.unsupported_patterns = [
            "mx_", "my_", "mz_", "mag_x", "mag_y", "mag_z",  # Magnetometer
            "audio_spectral_centroid", "audio_spectral_rolloff", "audio_spectral_bandwidth",  # Fake acoustic FFT on MAX9814
            "audio_mfcc_"  # Fake acoustic MFCC on MAX9814 amplitude
        ]
        
        self.supported_modalities = {
            'vibration': ['acc_', 'gyro_'],
            'acoustic_amplitude': ['audio_mean', 'audio_std', 'audio_min', 'audio_max', 'audio_rms', 'audio_iqr', 'audio_kurtosis', 'audio_skewness', 'audio_noise_floor', 'audio_crest_factor'],
            'thermal': ['temp_']
        }
        
        self.edge_dsp_requirements = {
            'acc_mag_wavelet_band_1_energy': 'DWT_DB4',
            'gyro_mag_harmonic_ratio': 'FFT_WINDOWED'
        }

    def is_feature_supported(self, feature_name):
        """
        Returns True if feature is physically valid for the hardware node, False otherwise.
        """
        for pattern in self.unsupported_patterns:
            if pattern in feature_name:
                return False
        return True

    def validate_feature_set(self, feature_list):
        """
        Validates feature list against hardware capabilities.
        Returns (is_valid, unsupported_features_found).
        """
        invalid_features = [f for f in feature_list if not self.is_feature_supported(f)]
        return (len(invalid_features) == 0, invalid_features)
