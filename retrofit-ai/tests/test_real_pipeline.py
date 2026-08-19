import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as np
import pandas as pd
from src.preprocessing.real_sensor_prep import preprocess_real_sensor_dataframe
from src.inference.inference import RetroFitInferencePipeline

class TestRealRetroFitPipeline(unittest.TestCase):
    def setUp(self):
        self.csv_path = r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv"
        self.models_dir = "data/models"

    def test_01_real_preprocessing(self):
        df_raw = pd.read_csv(self.csv_path)
        df_clean = preprocess_real_sensor_dataframe(df_raw, calibrate_gyro=True)
        
        self.assertNotIn('mx', df_clean.columns)
        self.assertNotIn('my', df_clean.columns)
        self.assertNotIn('mz', df_clean.columns)
        self.assertIn('acc_mag', df_clean.columns)
        self.assertIn('gyro_mag', df_clean.columns)
        self.assertIn('gx_cal', df_clean.columns)

    def test_02_real_inference_pipeline(self):
        features_csv = "data/features/real_features.csv"
        df_features = pd.read_csv(features_csv)
        pipeline = RetroFitInferencePipeline(models_dir=self.models_dir)
        
        out = pipeline.predict_window(df_features.iloc[[0]])
        self.assertIn('status', out)
        self.assertIn('similarity', out)
        self.assertIn('behavior_drift', out)
        self.assertIn('confidence', out)
        self.assertIn(out['status'], ['KNOWN_NORMAL_STATE', 'UNKNOWN_UNSEEN_BEHAVIOR'])

if __name__ == '__main__':
    unittest.main()
