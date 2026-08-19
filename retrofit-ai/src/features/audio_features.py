import numpy as np
from src.features.statistical import compute_statistical_features
from src.features.temporal import compute_temporal_features

def extract_audio_features(audio_data, fs_audio=200, enable_mfcc=False, is_amplitude_summary=True):
    """
    Extracts acoustic features for MAX9814 amplitude summary or raw waveform audio signals.
    When inputs are MAX9814 scalar amplitude summaries (sound_peak / sound_volts), extracts ONLY
    physically valid amplitude/noise-floor statistics without fabricating fake acoustic FFT/spectral features.
    """
    features = {}
    if audio_data is None or len(audio_data) == 0:
        return features
        
    # Statistical features (mean, std, min, max, RMS, peak-to-peak, kurtosis, skewness, IQR)
    features.update(compute_statistical_features(audio_data, prefix='audio'))
    
    # Temporal features (crest factor, impulse factor, margin factor)
    features.update(compute_temporal_features(audio_data, prefix='audio'))
    
    # Noise Floor Estimate (10th percentile of amplitude envelope)
    features['audio_noise_floor'] = float(np.percentile(audio_data, 10))
    
    # ONLY calculate FFT/MFCC if audio_data is true high-speed raw audio waveform (fs_audio >= 8000 Hz)
    if not is_amplitude_summary and fs_audio >= 8000:
        try:
            fft_vals = np.abs(np.fft.rfft(audio_data))
            freqs = np.fft.rfftfreq(len(audio_data), 1.0 / fs_audio)
            
            mel_bands = [(50, 500), (500, 1500), (1500, 4000), (4000, 7500)]
            for i, (f_low, f_high) in enumerate(mel_bands):
                mask = (freqs >= f_low) & (freqs < f_high)
                band_energy = np.sum(fft_vals[mask]**2) + 1e-12
                features[f"audio_mfcc_{i+1}"] = float(np.log(band_energy))
        except Exception:
            pass
            
    return features
