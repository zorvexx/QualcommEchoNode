import os
import pandas as pd
from src.edge_sim.interfaces import ISensorProvider

class CSVSensorStreamProvider(ISensorProvider):
    """
    Streams raw sensor samples from a recorded CSV file simulating hardware sensor reads.
    Magnetometer columns (mx, my, mz) are strictly IGNORED.
    """
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Sensor dataset file not found at: {csv_path}")
            
        self.df_raw = pd.read_csv(csv_path)
        self.current_idx = 0
        self.total_samples = len(self.df_raw)
        print(f"[EDGE SIMULATOR SENSORS] Loaded {self.total_samples} raw sensor samples from {csv_path}.")

    def is_data_available(self):
        return self.current_idx < self.total_samples

    def read_sample(self):
        if not self.is_data_available():
            return None
            
        row = self.df_raw.iloc[self.current_idx]
        self.current_idx += 1
        
        # Read MPU6050, MAX9814, MLX90614 readings (Ignore magnetometer mx, my, mz)
        sample = {
            'timestamp_ms': float(row.get('timestamp_ms', self.current_idx * 5.0)),
            'ax': float(row.get('ax', 0.0)),
            'ay': float(row.get('ay', 0.0)),
            'az': float(row.get('az', 9.8)),
            'gx': float(row.get('gx', 0.0)),
            'gy': float(row.get('gy', 0.0)),
            'gz': float(row.get('gz', 0.0)),
            'sound_peak': float(row.get('sound_peak', 0.0)),
            'sound_volts': float(row.get('sound_volts', 0.0)),
            'ir_ambient_c': float(row.get('ir_ambient_c', 25.0)),
            'ir_object_c': float(row.get('ir_object_c', 25.0))
        }
        return sample
