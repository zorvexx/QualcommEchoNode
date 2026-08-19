import numpy as np
from scipy import stats

def compute_spectral_features(x, fs, bands=None, prefix=''):
    """
    Computes FFT-based spectral features:
    dominant frequency, spectral energy, spectral centroid, spectral bandwidth,
    spectral roll-off, spectral entropy, spectral flatness, spectral flux,
    harmonic ratio, and configurable sub-band energies.
    """
    if len(x) == 0:
        return {}
    
    # FFT calculation
    fft_vals = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    
    psd = fft_vals ** 2
    psd_sum = np.sum(psd) + 1e-12
    psd_norm = psd / psd_sum
    
    dominant_freq = float(freqs[np.argmax(fft_vals)])
    spectral_energy = float(psd_sum)
    spectral_centroid = float(np.sum(freqs * psd_norm))
    spectral_bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm)))
    
    # Spectral Roll-off (85%)
    cum_psd = np.cumsum(psd_norm)
    roll_off_idx = np.where(cum_psd >= 0.85)[0]
    spectral_rolloff = float(freqs[roll_off_idx[0]]) if len(roll_off_idx) > 0 else float(freqs[-1])
    
    # Spectral Entropy
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    
    # Spectral Flatness (Geometric Mean / Arithmetic Mean)
    geom_mean = float(np.exp(np.mean(np.log(psd + 1e-12))))
    arith_mean = float(np.mean(psd))
    spectral_flatness = float(geom_mean / (arith_mean + 1e-12))
    
    # Harmonic ratio approximation (Peak / Sum)
    harmonic_ratio = float(np.max(fft_vals) / (np.sum(fft_vals) + 1e-12))
    
    pfx = f"{prefix}_" if prefix else ""
    features = {
        f"{pfx}dominant_freq": dominant_freq,
        f"{pfx}spectral_energy": spectral_energy,
        f"{pfx}spectral_centroid": spectral_centroid,
        f"{pfx}spectral_bandwidth": spectral_bandwidth,
        f"{pfx}spectral_rolloff": spectral_rolloff,
        f"{pfx}spectral_entropy": spectral_entropy,
        f"{pfx}spectral_flatness": spectral_flatness,
        f"{pfx}harmonic_ratio": harmonic_ratio
    }
    
    # Sub-band energies
    if bands is not None:
        for idx, (f_low, f_high) in enumerate(bands):
            mask = (freqs >= f_low) & (freqs < f_high)
            band_energy = float(np.sum(psd[mask]))
            features[f"{pfx}band_energy_{idx+1}_{f_low}_{f_high}hz"] = band_energy
            
    return features
