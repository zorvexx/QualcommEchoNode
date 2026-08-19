import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
from src.hardware.capabilities import HardwareCapabilityRegistry
from src.training.validation_gate import validate_hardware_feature_compatibility

class TestHardwareCapabilitiesGate(unittest.TestCase):
    def setUp(self):
        self.registry = HardwareCapabilityRegistry()

    def test_01_hardware_capability_registry(self):
        # Valid features must pass
        self.assertTrue(self.registry.is_feature_supported("acc_y_rms"))
        self.assertTrue(self.registry.is_feature_supported("gyro_mag_mean"))
        self.assertTrue(self.registry.is_feature_supported("audio_noise_floor"))
        self.assertTrue(self.registry.is_feature_supported("temp_slope"))
        
        # Invalid features must fail
        self.assertFalse(self.registry.is_feature_supported("audio_spectral_centroid"))
        self.assertFalse(self.registry.is_feature_supported("mx_mean"))
        self.assertFalse(self.registry.is_feature_supported("audio_mfcc_1"))

    def test_02_validation_gate_rejection(self):
        invalid_set = ["acc_y_rms", "audio_spectral_centroid"]
        with self.assertRaises(ValueError):
            validate_hardware_feature_compatibility(invalid_set, strict=True)

    def test_03_validation_gate_success(self):
        valid_set = ["acc_y_rms", "gyro_mag_mean", "audio_noise_floor", "temp_slope"]
        res = validate_hardware_feature_compatibility(valid_set, strict=True)
        self.assertTrue(res)

if __name__ == '__main__':
    unittest.main()
