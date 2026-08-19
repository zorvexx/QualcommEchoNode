import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import pandas as pd
from src.edge_sim.edge_runner import run_uno_q_edge_simulation
from src.inference.inference import RetroFitInferencePipeline

class TestUnoQEdgeSimulator(unittest.TestCase):
    def setUp(self):
        self.csv_path = r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv"

    def test_simulator_execution_and_match(self):
        # Run edge simulator on real sensor dataset
        res = run_uno_q_edge_simulation(csv_path=self.csv_path, machine_id="DEV_01")
        
        self.assertGreater(res['window_count'], 0)
        self.assertIn('avg_total_ms', res['latencies'])
        self.assertLess(res['latencies']['avg_total_ms'], 1024.0) # Fits within 1024 ms hop budget
        self.assertGreater(res['header_size_kb'], 0.0)
        
        # Verify simulator outputs match RetroFitInferencePipeline direct outputs
        direct_pipeline = RetroFitInferencePipeline(machine_id="DEV_01")
        sim_out = res['outputs'][0]
        
        self.assertEqual(sim_out['machine_id'], direct_pipeline.fingerprint.get('machine_id', 'DEV_01'))
        self.assertIn(sim_out['status'], ['KNOWN_NORMAL_STATE', 'UNKNOWN_UNSEEN_BEHAVIOR', 'CRITICAL_ANOMALY'])

if __name__ == '__main__':
    unittest.main()
