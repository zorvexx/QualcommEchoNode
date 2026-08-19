import numpy as np
from scipy import signal

def preprocess_vibration(accel_data, fs=1000, highpass_cutoff=3.0, lowpass_cutoff=450.0):
    """
    Applies high-pass Butterworth filter (DC/gravity removal) and low-pass anti-aliasing filter.
    accel_data: np.ndarray of shape (N, 3) or (N,)
    """
    if accel_data is None or len(accel_data) == 0:
        return accel_data
    
    nyq = 0.5 * fs
    # High-pass filter
    if highpass_cutoff > 0 and nyq > highpass_cutoff:
        b_hp, a_hp = signal.butter(3, highpass_cutoff / nyq, btype='high')
        accel_data = signal.filtfilt(b_hp, a_hp, accel_data, axis=0)
        
    # Low-pass filter
    if lowpass_cutoff > 0 and nyq > lowpass_cutoff:
        b_lp, a_lp = signal.butter(4, lowpass_cutoff / nyq, btype='low')
        accel_data = signal.filtfilt(b_lp, a_lp, accel_data, axis=0)
        
    return accel_data
