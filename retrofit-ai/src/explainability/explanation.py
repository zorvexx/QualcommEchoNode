import json
import os

def generate_explanation_json(modality_contrib, top_features, feature_contribs, output_path="data/reports/feature_contributions.json"):
    """
    Saves explainability output to data/reports/feature_contributions.json.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        'modality_contribution': modality_contrib,
        'top_features': top_features,
        'feature_contributions': feature_contribs
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    return data

def generate_machine_personality(fingerprint, feature_contribs):
    """
    Generates rule-based machine personality text based on learned statistics.
    """
    vib_mod = feature_contribs.get('acc_mag_rms', 0.0)
    temp_mod = feature_contribs.get('temp_slope', 0.0)
    audio_mod = feature_contribs.get('audio_spectral_entropy', 0.0)
    
    vib_state = "highly stable" if vib_mod < 0.5 else "moderately variable"
    audio_state = "low broadband noise" if audio_mod < 0.5 else "variable acoustic timbre"
    temp_state = "gradually increasing under load" if temp_mod >= 0 else "thermally constant"
    
    personality = {
        'vibration': vib_state,
        'acoustic': audio_state,
        'thermal': temp_state,
        'operating_rhythm': 'highly consistent'
    }
    return personality
