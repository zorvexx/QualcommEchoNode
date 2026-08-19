import unittest
import os
import numpy as np
import pandas as pd
import yaml

from scripts.generate_demo_data import generate_demo_dataset
from src.features.extractor import extract_features_from_dataframe
from src.selection.selector import select_top_features
from src.states.state_discovery import OperatingStateDiscoverer
from src.models.autoencoder import AutoencoderAnomalyDetector
from src.behavior.similarity import calculate_behavioral_similarity

class TestRetroFitPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.demo_csv = "data/raw/test_demo.csv"
        generate_demo_dataset(cls.demo_csv, duration_seconds=30)
        with open("config.yaml", "r") as f:
            cls.config = yaml.safe_load(f)

    def test_01_feature_extraction(self):
        df_raw = pd.read_csv(self.demo_csv)
        df_feat = extract_features_from_dataframe(df_raw, self.config)
        self.assertGreater(len(df_feat), 0)
        self.assertIn('acc_mag_rms', df_feat.columns)
        self.assertIn('audio_rms', df_feat.columns)

    def test_02_feature_selection(self):
        df_raw = pd.read_csv(self.demo_csv)
        df_feat = extract_features_from_dataframe(df_raw, self.config)
        feature_cols = [c for c in df_feat.columns if c not in ['timestamp', 'machine_id', 'session_id', 'operating_state', 'label']]
        selected, _ = select_top_features(df_feat[feature_cols], top_n=15)
        self.assertEqual(len(selected), 15)

    def test_03_state_discovery(self):
        df_raw = pd.read_csv(self.demo_csv)
        df_feat = extract_features_from_dataframe(df_raw, self.config)
        feature_cols = [c for c in df_feat.columns if c not in ['timestamp', 'machine_id', 'session_id', 'operating_state', 'label']][:10]
        state_model = OperatingStateDiscoverer(min_clusters=2, max_clusters=3).fit(df_feat[feature_cols])
        labels = state_model.predict(df_feat[feature_cols])
        self.assertEqual(len(labels), len(df_feat))

    def test_04_similarity_calc(self):
        sim_normal = calculate_behavioral_similarity(0.5, threshold=1.0)
        sim_drift = calculate_behavioral_similarity(2.0, threshold=1.0)
        self.assertEqual(sim_normal, 100.0)
        self.assertLess(sim_drift, 100.0)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.demo_csv):
            os.remove(cls.demo_csv)

if __name__ == '__main__':
    unittest.main()
