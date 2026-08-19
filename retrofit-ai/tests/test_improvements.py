import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import unittest
import numpy as np
import pandas as pd

from src.inference.inference import RetroFitInferencePipeline
from src.edge.export_cpp_header import export_isolation_forest_cpp_header
from src.alerts.escalation import AlertEscalationEngine
from src.communication.mqtt_client import RetroFitMQTTPublisher
from src.preprocessing.real_sensor_prep import load_or_compute_gyro_calibration
from src.features.audio_features import extract_audio_features
from src.replay.demo_runner import run_replay_demo

class TestRetroFitImprovements(unittest.TestCase):
    def setUp(self):
        self.csv_path = r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv"
        self.models_dir = "data/models"

    def test_01_multi_machine_inference_routing(self):
        # Test machine routing fallback
        pipeline = RetroFitInferencePipeline(machine_id="NON_EXISTENT_MACHINE", models_dir=self.models_dir)
        self.assertEqual(pipeline.models_dir, self.models_dir)

    def test_02_cpp_header_exporter(self):
        res = export_isolation_forest_cpp_header(models_dir=self.models_dir, output_header="data/test_header.h")
        self.assertIsNotNone(res)
        self.assertTrue(os.path.exists("data/test_header.h"))
        self.assertGreater(res['size_kb'], 0.0)
        self.assertEqual(res['n_trees'], 100)

    def test_03_persistent_alert_cooldown_history(self):
        m_cfg = {'machine_id': 'TEST_PERSIST_01', 'machine_name': 'Test Unit', 'operator_phone': '+15550199'}
        hist_path = "data/machines/TEST_PERSIST_01/alert_history.json"
        if os.path.exists(hist_path):
            os.remove(hist_path)
            
        engine = AlertEscalationEngine(confirmation_count=1, min_confidence=70.0, cooldown_period_sec=300)
        
        # 1st observation -> Call placed, timestamp persisted to disk
        a1 = engine.evaluate_observation('CRITICAL_ANOMALY', 0.5, 45.0, 95.0, m_cfg)
        self.assertTrue(a1['twilio_call_triggered'])
        self.assertTrue(os.path.exists(hist_path))
        
        # 2nd observation -> Call suppressed by persistent cooldown
        engine_restarted = AlertEscalationEngine(confirmation_count=1, min_confidence=70.0, cooldown_period_sec=300)
        a2 = engine_restarted.evaluate_observation('CRITICAL_ANOMALY', 0.5, 45.0, 95.0, m_cfg)
        self.assertFalse(a2['twilio_call_triggered'])
        self.assertIn('suppressed by cooldown', a2['message'])

    def test_04_non_blocking_mqtt_queue(self):
        publisher = RetroFitMQTTPublisher()
        res = publisher.publish_telemetry({'machine_id': 'TEST_ASYNC', 'similarity': 99.0, 'status': 'KNOWN_NORMAL_STATE'})
        self.assertEqual(res['status'], 'ENQUEUED_NON_BLOCKING')

    def test_05_calibration_stability_validation(self):
        cal_path = "data/test_calibration/test_gyro_cal_qual.json"
        if os.path.exists(cal_path):
            os.remove(cal_path)
            
        df_stable = pd.DataFrame({
            'ax': [0]*100, 'ay': [0]*100, 'az': [9.8]*100,
            'gx': np.random.normal(100, 5, 100),
            'gy': np.random.normal(300, 5, 100),
            'gz': np.random.normal(100, 5, 100)
        })
        
        cal = load_or_compute_gyro_calibration(df_stable, cal_file=cal_path, max_noise_std_threshold=500.0)
        self.assertIn('zro_gx', cal)
        
        qual_path = "data/test_calibration/calibration_quality.json"
        self.assertTrue(os.path.exists(qual_path))
        with open(qual_path, 'r') as f:
            qual = json.load(f)
            self.assertTrue(qual['is_stable'])

    def test_06_max9814_amplitude_sanitization(self):
        # Amplitude summary data sampled at 200 Hz
        sound_volts = np.random.normal(1.45, 0.2, 410)
        feats = extract_audio_features(sound_volts, fs_audio=200, is_amplitude_summary=True)
        
        self.assertIn('audio_mean', feats)
        self.assertIn('audio_noise_floor', feats)
        self.assertNotIn('audio_mfcc_1', feats) # Verify fake FFT spectrum is NOT computed

if __name__ == '__main__':
    unittest.main()
