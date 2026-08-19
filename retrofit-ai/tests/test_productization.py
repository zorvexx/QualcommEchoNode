import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as np
import pandas as pd

from src.registration.manager import MachineRegistrationManager
from src.behavior.learning import AdaptiveBaselineLearner
from src.preprocessing.real_sensor_prep import load_or_compute_gyro_calibration, preprocess_real_sensor_dataframe
from src.alerts.escalation import AlertEscalationEngine
from src.alerts.twilio_voice import TwilioVoiceNotifier
from src.communication.mqtt_client import RetroFitMQTTPublisher
from src.edge.actuation import HardwareActuator
from src.replay.demo_runner import run_replay_demo

class TestRetroFitProductization(unittest.TestCase):
    def setUp(self):
        self.csv_path = r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv"
        self.test_m_id = "TEST_MACHINE_99"

    def test_01_machine_registration_and_lifecycle(self):
        mgr = MachineRegistrationManager(base_dir="data/test_machines")
        cfg = mgr.register_machine(self.test_m_id, "Test Motor", "Spindle", location="Bench A", operator_phone="+15550199")
        self.assertEqual(cfg['lifecycle_state'], 'REGISTERED')
        
        cfg_updated = mgr.update_lifecycle_state(self.test_m_id, "MONITORING")
        self.assertEqual(cfg_updated['lifecycle_state'], 'MONITORING')

    def test_02_adaptive_learning_freeze_safeguards(self):
        learner = AdaptiveBaselineLearner(enabled=False)
        # Verify disabled by default
        res = learner.process_observation('KNOWN_NORMAL_STATE', 0.05, [1.0, 2.0], 99.0)
        self.assertFalse(res['adapted'])
        
        # Verify non-normal status is REJECTED even if enabled
        learner_enabled = AdaptiveBaselineLearner(enabled=True)
        res_reject = learner_enabled.process_observation('CRITICAL_ANOMALY', 0.85, [1.0, 2.0], 99.0)
        self.assertFalse(res_reject['adapted'])
        self.assertIn('Rejected', res_reject['reason'])

    def test_03_dedicated_gyro_calibration(self):
        cal_path = "data/test_calibration/test_gyro_cal.json"
        if os.path.exists(cal_path):
            os.remove(cal_path)
            
        df_raw = pd.DataFrame({
            'ax': [0, 0], 'ay': [0, 0], 'az': [9.8, 9.8],
            'gx': [100.0, 102.0], 'gy': [330.0, 332.0], 'gz': [110.0, 112.0]
        })
        cal = load_or_compute_gyro_calibration(df_raw, cal_file=cal_path)
        self.assertIn('zro_gx', cal)
        self.assertEqual(cal['zro_gx'], 101.0)
        
        df_clean = preprocess_real_sensor_dataframe(df_raw, calibrate_gyro=True, cal_file=cal_path)
        self.assertIn('gx_cal', df_clean.columns)
        self.assertAlmostEqual(df_clean['gx_cal'].iloc[0], -1.0)

    def test_04_alert_escalation_confirmation_and_cooldown(self):
        hist_path = "data/machines/M1/alert_history.json"
        if os.path.exists(hist_path):
            os.remove(hist_path)
            
        engine = AlertEscalationEngine(confirmation_count=3, min_confidence=70.0, cooldown_period_sec=300)
        cfg = {'machine_name': 'Test Machine', 'machine_id': 'M1', 'location': 'Lab', 'operator_phone': '+15550199'}
        
        # 1st critical -> Warning logged, no Twilio call
        a1 = engine.evaluate_observation('CRITICAL_ANOMALY', 0.4, 25.0, 95.0, cfg)
        self.assertFalse(a1['confirmed_critical'])
        
        # 2nd critical -> Warning logged
        a2 = engine.evaluate_observation('CRITICAL_ANOMALY', 0.4, 30.0, 95.0, cfg)
        self.assertFalse(a2['confirmed_critical'])
        
        # 3rd critical -> Confirmed critical!
        a3 = engine.evaluate_observation('CRITICAL_ANOMALY', 0.5, 45.0, 95.0, cfg)
        self.assertTrue(a3['confirmed_critical'])
        self.assertTrue(a3['twilio_call_triggered'])

    def test_05_twilio_mock_mode_safety(self):
        # Without credentials, TwilioVoiceNotifier must safely run in Mock Mode
        notifier = TwilioVoiceNotifier()
        res = notifier.trigger_voice_alert("+15550199", "Test Machine", "M1", "Lab", 45.0, ['vibration', 'temp'])
        self.assertEqual(res['mode'], 'MOCK')
        self.assertEqual(res['status'], 'SUCCESS')

    def test_06_hardware_actuation_mapping(self):
        actuator = HardwareActuator()
        cmd_norm = actuator.get_actuation_command('KNOWN_NORMAL_STATE')
        self.assertEqual(cmd_norm['led_ring'], 'SOLID_GREEN')
        
        cmd_crit = actuator.get_actuation_command('CRITICAL_ANOMALY')
        self.assertEqual(cmd_crit['led_ring'], 'FLASHING_RED')
        self.assertTrue(cmd_crit['buzzer_active'])

    def test_07_mqtt_publisher_payload(self):
        publisher = RetroFitMQTTPublisher()
        res = publisher.publish_telemetry({'machine_id': 'TEST_01', 'similarity': 98.5, 'status': 'KNOWN_NORMAL_STATE'})
        self.assertIn(res['status'], ['ENQUEUED_NON_BLOCKING', 'PUBLISHED', 'LOCAL_FALLBACK'])
        self.assertEqual(res['payload']['similarity'], 98.5)

if __name__ == '__main__':
    unittest.main()
