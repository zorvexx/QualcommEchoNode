import unittest
import os
import shutil
import numpy as np
import pandas as pd

from src.onboarding.baseline_learning import AdaptiveBaselineOnboarder
from src.registration.manager import MachineRegistrationManager

class TestAdaptiveOnboarding(unittest.TestCase):
    def setUp(self):
        self.test_dir = "data/test_onboarding_machines"
        os.makedirs(self.test_dir, exist_ok=True)
        self.onboarder = AdaptiveBaselineOnboarder(
            min_duration_sec=10.0,  # Shortened for quick test execution
            min_windows=15,
            max_gap_ms=1000.0,
            max_contamination_pct=15.0
        )
        self.reg_mgr = MachineRegistrationManager(base_dir=self.test_dir)
        self.machine_id = "TEST_ADAPTIVE_01"
        self.reg_mgr.register_machine(self.machine_id, "Test Machine", "Motor")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_features(self, n_windows=20, duration_sec=15.0, add_disturbance=False, thermal_warmup=True):
        timestamps = np.linspace(0.0, duration_sec * 1000.0, n_windows)
        
        # Build 30 dummy physically valid features
        data = {
            'timestamp': timestamps,
            'machine_id': self.machine_id,
            'session_label': 'idle_01',
            'operating_state': 'IDLE_NORMAL',
            'label': 0,
            'eff_fs_win': 100.0
        }
        
        for k in range(30):
            data[f'feature_{k}'] = np.random.normal(loc=0.0, scale=1.0, size=n_windows)
            
        data['acc_mag_mean'] = np.random.normal(loc=13500.0, scale=50.0, size=n_windows)
        data['acc_mag_std'] = np.random.normal(loc=100.0, scale=10.0, size=n_windows)
        
        if thermal_warmup:
            # Add gradual thermal warmup
            data['temp_object_mean'] = np.linspace(26.0, 26.8, n_windows)
            data['temp_ambient_mean'] = np.linspace(25.7, 26.4, n_windows)
            
        if add_disturbance:
            # Inject severe disturbance into 5 windows (> 25% contamination)
            for idx in [2, 3, 4, 5, 6]:
                data['acc_mag_std'][idx] = 10000.0
                
        return pd.DataFrame(data)

    def test_insufficient_baseline_fails(self):
        """Test 1: Insufficient baseline sample/window count raises ValueError guard."""
        df_insufficient = self._create_mock_features(n_windows=5, duration_sec=3.0)
        with self.assertRaises(ValueError) as ctx:
            self.onboarder.onboard_machine_baseline(
                df_insufficient,
                self.machine_id,
                output_dir=self.test_dir
            )
        self.assertIn("INSUFFICIENT BASELINE DATA", str(ctx.exception))

    def test_stable_baseline_passes(self):
        """Test 2: Stable baseline successfully completes onboarding into MONITORING state."""
        df_stable = self._create_mock_features(n_windows=20, duration_sec=15.0)
        meta = self.onboarder.onboard_machine_baseline(
            df_stable,
            self.machine_id,
            output_dir=self.test_dir
        )
        self.assertEqual(meta['onboarding_status'], 'BASELINE_READY')
        self.assertEqual(meta['lifecycle_state'], 'MONITORING')
        self.assertGreater(meta['baseline_stability_score'], 80.0)

    def test_disturbed_baseline_fails_stability(self):
        """Test 3: Disturbed baseline fails stability check with BASELINE_NOT_READY."""
        df_disturbed = self._create_mock_features(n_windows=20, duration_sec=15.0, add_disturbance=True)
        with self.assertRaises(ValueError) as ctx:
            self.onboarder.onboard_machine_baseline(
                df_disturbed,
                self.machine_id,
                output_dir=self.test_dir
            )
        self.assertIn("BASELINE NOT READY", str(ctx.exception))
        
        cfg = self.reg_mgr.get_machine_config(self.machine_id)
        self.assertEqual(cfg['lifecycle_state'], 'BASELINE_NOT_READY')

    def test_thermal_drift_handled_gracefully(self):
        """Test 4: Gradual thermal warmup is recognized as normal operational transition."""
        df_thermal = self._create_mock_features(n_windows=20, duration_sec=15.0, thermal_warmup=True)
        stability_res = self.onboarder.analyze_baseline_stability(df_thermal)
        self.assertTrue(stability_res['is_stable'])
        self.assertGreater(stability_res['thermal_slope_c_per_sec'], 0.0)

    def test_onboarding_lifecycle_completion(self):
        """Test 5: Machine lifecycle transitions cleanly to MONITORING."""
        df_stable = self._create_mock_features(n_windows=20, duration_sec=15.0)
        self.onboarder.onboard_machine_baseline(
            df_stable,
            self.machine_id,
            output_dir=self.test_dir
        )
        cfg = self.reg_mgr.get_machine_config(self.machine_id)
        self.assertEqual(cfg['lifecycle_state'], 'MONITORING')

if __name__ == "__main__":
    unittest.main()
