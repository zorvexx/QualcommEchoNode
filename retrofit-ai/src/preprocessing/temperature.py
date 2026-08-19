import numpy as np
from scipy import signal

def preprocess_temperature(temp_data, window_length=11, polyorder=2):
    """
    Smooths thermal signal using Savitzky-Golay filter if window length allows.
    """
    if temp_data is None or len(temp_data) == 0:
        return temp_data
    
    if len(temp_data) >= window_length:
        return signal.savgol_filter(temp_data, window_length, polyorder)
    return temp_data
