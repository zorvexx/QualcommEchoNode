import numpy as np

def compute_modality_contributions(feature_names, residuals_or_deviations):
    """
    Groups per-feature residual/deviation errors into modality buckets:
    vibration, audio, temperature, magnetic.
    Returns normalized contributions summing to 1.0.
    """
    modality_sums = {'vibration': 0.0, 'audio': 0.0, 'temperature': 0.0, 'magnetic': 0.0}
    
    for fname, err in zip(feature_names, residuals_or_deviations):
        val = abs(float(err))
        if fname.startswith(('acc_', 'gyro_')):
            modality_sums['vibration'] += val
        elif fname.startswith('audio_'):
            modality_sums['audio'] += val
        elif fname.startswith('temp_'):
            modality_sums['temperature'] += val
        elif fname.startswith('mag_'):
            modality_sums['magnetic'] += val
        else:
            modality_sums['vibration'] += val
            
    total_val = sum(modality_sums.values()) + 1e-9
    contributions = {k: round(v / total_val, 4) for k, v in modality_sums.items()}
    return contributions
