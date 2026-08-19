import pandas as pd
import numpy as np

def synchronize_multimodal_df(df):
    """
    Ensures missing optional columns exist with NaN or defaults, and fills forward/backward.
    """
    expected_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'audio', 'temperature']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    # Forward and backward fill missing values
    df = df.ffill().bfill().fillna(0.0)
    return df
