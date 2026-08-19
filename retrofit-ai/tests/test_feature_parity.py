import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import unittest
import numpy as np
import pandas as pd

from src.preprocessing.real_sensor_prep import preprocess_real_sensor_dataframe
from src.features.vibration_features import extract_vibration_features
from src.features.audio_features import extract_audio_features
from src.features.temperature_features import extract_temperature_features
from src.inference.inference import RetroFitInferencePipeline
from src.edge_sim.edge_runner import run_uno_q_edge_simulation

class TestEdgePythonFeatureParity(unittest.TestCase):
    def setUp(self):
        self.csv_path = r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv"
        with open("data/models/selected_features.json", "r") as f:
            self.selected_features = json.load(f)

    def test_feature_parity_numeric_equivalence(self):
        df_raw = pd.read_csv(self.csv_path)
        win_raw = df_raw.iloc[:410]
        
        # 1. Reference Python feature extraction
        win_clean = preprocess_real_sensor_dataframe(win_raw, calibrate_gyro=True)
        accel_data = win_clean[['ax', 'ay', 'az']].values
        gyro_data = win_clean[['gx_cal', 'gy_cal', 'gz_cal']].values
        
        f_vib = extract_vibration_features(accel_data, gyro_data, fs_accel=200, fs_gyro=200)
        f_aud = extract_audio_features(win_clean['sound_volts'].values, fs_audio=200, is_amplitude_summary=True)
        f_tmp = extract_temperature_features(win_clean['ir_object_c'].values, baseline_temp=float(win_clean['ir_ambient_c'].iloc[0]))
        
        ref_dict = {}
        ref_dict.update(f_vib)
        ref_dict.update(f_aud)
        ref_dict.update(f_tmp)
        
        # 2. Edge Simulator feature extraction
        sim_res = run_uno_q_edge_simulation(csv_path=self.csv_path, machine_id="DEV_01")
        sim_out = sim_res['outputs'][0]
        
        # 3. Direct Pipeline evaluation
        pipeline = RetroFitInferencePipeline(machine_id="DEV_01")
        df_ref_win = pd.DataFrame([ref_dict])
        for f in self.selected_features:
            if f not in df_ref_win.columns:
                df_ref_win[f] = 0.0
                
        direct_out = pipeline.predict_window(df_ref_win)
        
        # Numeric equivalence assertions
        self.assertAlmostEqual(direct_out['anomaly_score'], sim_out['anomaly_score'], places=5)
        self.assertAlmostEqual(direct_out['similarity'], sim_out['similarity'], places=1)
        self.assertEqual(direct_out['status'], sim_out['status'])

if __name__ == '__main__':
    unittest.main()
