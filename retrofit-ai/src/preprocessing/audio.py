import numpy as np
from scipy import signal

def preprocess_audio(audio_data, fs=16000, lowcut=50.0, highcut=7500.0):
    """
    Applies DC offset subtraction and bandpass filtering.
    """
    if audio_data is None or len(audio_data) == 0:
        return audio_data
    
    # DC offset removal
    audio_data = audio_data - np.mean(audio_data)
    
    nyq = 0.5 * fs
    if nyq > highcut:
        b, a = signal.butter(4, [lowcut / nyq, highcut / nyq], btype='band')
        audio_data = signal.filtfilt(b, a, audio_data)
        
    return audio_data
