import numpy as np

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

def compute_wavelet_features(x, wavelet='db4', level=3, prefix=''):
    """
    Computes compact wavelet packet energy ratio, band energy, and wavelet entropy using PyWavelets.
    """
    if not HAS_PYWT or len(x) == 0:
        return {}
    
    pfx = f"{prefix}_" if prefix else ""
    try:
        coeffs = pywt.wavedec(x, wavelet, level=level)
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = np.sum(energies) + 1e-12
        
        energy_ratios = [float(e / total_energy) for e in energies]
        
        # Wavelet entropy
        probs = np.array(energy_ratios) + 1e-12
        wavelet_entropy = float(-np.sum(probs * np.log2(probs)))
        
        feats = {f"{pfx}wavelet_entropy": wavelet_entropy}
        for idx, ratio in enumerate(energy_ratios):
            feats[f"{pfx}wavelet_band_{idx}_ratio"] = ratio
            feats[f"{pfx}wavelet_band_{idx}_energy"] = float(energies[idx])
            
        return feats
    except Exception:
        return {}
