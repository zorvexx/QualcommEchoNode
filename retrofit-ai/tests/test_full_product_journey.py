import unittest
import os
import shutil
import json
import numpy as np
import pandas as pd

from src.registration.manager import MachineRegistrationManager
from src.preprocessing.real_sensor_prep import load_or_compute_gyro_calibration
from src.onboarding.baseline_learning import AdaptiveBaselineOnboarder
from src.inference.inference import RetroFitInferencePipeline
from src.alerts.escalation import AlertEscalationEngine
from src.communication.mqtt_client import RetroFitMQTTPublisher

class TestFullProductJourney(unittest.TestCase):
    """
    End-to-End Product Journey Integration Test for RetroFit.
    Simulates complete machine lifecycle from registration to monitoring, anomaly escalation,
    explanation, actuation, persistence, and recovery.
    """
    def setUp(self):
        self.base_dir = "data/test_journey_machines"
        os.makedirs(self.base_dir, exist_ok=True)
        self.machine_id = "JOURNEY_PUMP_01"
        
        # Clean stale alert history file if exists
        stale_dir = os.path.join("data", "machines", self.machine_id)
        if os.path.exists(stale_dir):
            shutil.rmtree(stale_dir, ignore_errors=True)
            
        self.reg_mgr = MachineRegistrationManager(base_dir=self.base_dir)
        self.onboarder = AdaptiveBaselineOnboarder(
            min_duration_sec=10.0,
            min_windows=15,
            max_gap_ms=1000.0,
            max_contamination_pct=15.0
        )

    def tearDown(self):
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir, ignore_errors=True)
        stale_dir = os.path.join("data", "machines", self.machine_id)
        if os.path.exists(stale_dir):
            shutil.rmtree(stale_dir, ignore_errors=True)

    def _generate_synthetic_baseline(self, n_windows=20, duration_sec=15.0):
        data = {
            'timestamp': np.linspace(0.0, duration_sec * 1000.0, n_windows),
            'machine_id': self.machine_id,
            'session_label': 'baseline_01',
            'operating_state': 'IDLE_NORMAL',
            'label': 0,
            'eff_fs_win': 100.0
        }
        for k in range(30):
            data[f'feature_{k}'] = np.random.normal(loc=0.0, scale=0.5, size=n_windows)
        data['acc_mag_mean'] = np.random.normal(loc=13500.0, scale=30.0, size=n_windows)
        data['acc_mag_std'] = np.random.normal(loc=100.0, scale=10.0, size=n_windows)
        data['temp_object_mean'] = np.linspace(26.0, 26.5, n_windows)
        data['temp_ambient_mean'] = np.linspace(25.7, 26.1, n_windows)
        return pd.DataFrame(data)

    def test_complete_product_journey(self):
        # 1. Stage: REGISTER MACHINE
        cfg = self.reg_mgr.register_machine(
            machine_id=self.machine_id,
            machine_name="Cooling Water Pump",
            machine_type="Centrifugal Pump",
            operator_phone="+15550199"
        )
        self.assertEqual(cfg['lifecycle_state'], 'REGISTERED')
        
        # 2. Stage: CALIBRATION
        cal_res = load_or_compute_gyro_calibration()
        self.reg_mgr.update_lifecycle_state(self.machine_id, "CALIBRATION")
        self.assertIn('zro_gx', cal_res)
        
        # 3. Stage: BASELINE_NOT_READY (Sample Guard Rejection Test)
        df_short = self._generate_synthetic_baseline(n_windows=5, duration_sec=3.0)
        with self.assertRaises(ValueError):
            self.onboarder.onboard_machine_baseline(df_short, self.machine_id, output_dir=self.base_dir)
            
        cfg_nr = self.reg_mgr.get_machine_config(self.machine_id)
        self.assertEqual(cfg_nr['lifecycle_state'], 'BASELINE_NOT_READY')
        
        # 4. Stage: LEARNING & BASELINE_READY & FINGERPRINT_CREATED & MONITORING
        df_valid = self._generate_synthetic_baseline(n_windows=20, duration_sec=15.0)
        meta = self.onboarder.onboard_machine_baseline(df_valid, self.machine_id, output_dir=self.base_dir)
        self.assertEqual(meta['onboarding_status'], 'BASELINE_READY')
        self.assertEqual(meta['lifecycle_state'], 'MONITORING')
        
        # Verify machine-specific storage
        mach_dir = os.path.join(self.base_dir, self.machine_id)
        self.assertTrue(os.path.exists(os.path.join(mach_dir, "scaler.pkl")))
        self.assertTrue(os.path.exists(os.path.join(mach_dir, "anomaly_model.pkl")))
        self.assertTrue(os.path.exists(os.path.join(mach_dir, "machine_fingerprint.json")))
        self.assertTrue(os.path.exists(os.path.join(mach_dir, "onboarding_metadata.json")))
        
        # 5. Stage: INFERENCE & THREE-TIER DECISION ENGINE
        pipeline = RetroFitInferencePipeline(machine_id=self.machine_id, base_machines_dir=self.base_dir)
        
        # A) Test KNOWN_NORMAL_STATE
        norm_sample = df_valid.iloc[[0]].copy()
        res_norm = pipeline.predict_window(norm_sample)
        self.assertIn(res_norm['status'], ['KNOWN_NORMAL_STATE', 'UNKNOWN_UNSEEN_BEHAVIOR'])
        self.assertIn('similarity', res_norm)
        self.assertIn('confidence', res_norm)
        
        # B) Test UNKNOWN_UNSEEN_BEHAVIOR & CRITICAL_ANOMALY & EXPLANATION
        anom_sample = norm_sample.copy()
        for col in meta['selected_features']:
            anom_sample[col] = 1000.0 # Force extreme outlier
            
        res_anom = pipeline.predict_window(anom_sample)
        self.assertEqual(res_anom['status'], 'CRITICAL_ANOMALY')
        self.assertIn('vibration', res_anom['modality_contribution'])
        self.assertIn('top_features', res_anom)
        
        # 6. Stage: ALERT / TWILIO MOCK & ALERT COOLDOWN & MQTT & RESTART PERSISTENCE
        escalation = AlertEscalationEngine(confirmation_count=3, cooldown_period_sec=300)
        action_1 = escalation.evaluate_observation(
            status=res_anom['status'],
            anomaly_score=res_anom['anomaly_score'],
            drift_score=res_anom['behavior_drift'],
            confidence=95.0,
            machine_config=cfg,
            top_features=res_anom['top_features']
        )
        action_2 = escalation.evaluate_observation(
            status=res_anom['status'],
            anomaly_score=res_anom['anomaly_score'],
            drift_score=res_anom['behavior_drift'],
            confidence=95.0,
            machine_config=cfg,
            top_features=res_anom['top_features']
        )
        action_res = escalation.evaluate_observation(
            status=res_anom['status'],
            anomaly_score=res_anom['anomaly_score'],
            drift_score=res_anom['behavior_drift'],
            confidence=95.0,
            machine_config=cfg,
            top_features=res_anom['top_features']
        )
        self.assertTrue(action_res['twilio_call_triggered']) # 3rd consecutive critical triggers alert
        
        # Verify Cooldown Persistence (4th observation during cooldown)
        action_cooldown = escalation.evaluate_observation(
            status=res_anom['status'],
            anomaly_score=res_anom['anomaly_score'],
            drift_score=res_anom['behavior_drift'],
            confidence=95.0,
            machine_config=cfg,
            top_features=res_anom['top_features']
        )
        self.assertFalse(action_cooldown['twilio_call_triggered']) # Cooldown enforced
        
        # 7. Stage: MQTT TELEMETRY
        mqtt_publisher = RetroFitMQTTPublisher(broker_host="localhost")
        mqtt_success = mqtt_publisher.publish_telemetry(res_anom)
        self.assertTrue(mqtt_success)
        
        # 8. Stage: RECOVERY BACK TO NORMAL
        res_recovered = pipeline.predict_window(norm_sample)
        self.assertIn(res_recovered['status'], ['KNOWN_NORMAL_STATE', 'UNKNOWN_UNSEEN_BEHAVIOR'])
        self.assertGreater(res_recovered['similarity'], 50.0)

if __name__ == "__main__":
    unittest.main()
