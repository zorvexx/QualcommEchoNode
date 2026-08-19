import numpy as np

def extract_temperature_features(temp_data, baseline_temp=25.0):
    """
    Extracts thermal temporal features: current temp, delta T, slope, rate of rise/fall, moving avg, std, stability.
    temp_data: 1D array of temperature samples in window
    """
    features = {}
    if temp_data is None or len(temp_data) == 0:
        return features
    
    current_temp = float(temp_data[-1])
    delta_temp = float(current_temp - baseline_temp)
    moving_avg = float(np.mean(temp_data))
    moving_std = float(np.std(temp_data))
    thermal_var = float(np.var(temp_data))
    
    # Temperature slope (rate of rise / fall)
    if len(temp_data) > 1:
        temp_slope = float(temp_data[-1] - temp_data[0])
        rate_of_rise = max(0.0, temp_slope)
        rate_of_fall = max(0.0, -temp_slope)
    else:
        temp_slope = 0.0
        rate_of_rise = 0.0
        rate_of_fall = 0.0
        
    thermal_stability = float(moving_std / (moving_avg + 1e-9))
    
    features.update({
        'temp_current': current_temp,
        'temp_baseline_relative': delta_temp,
        'temp_moving_avg': moving_avg,
        'temp_moving_std': moving_std,
        'temp_variance': thermal_var,
        'temp_slope': temp_slope,
        'temp_rate_of_rise': rate_of_rise,
        'temp_rate_of_fall': rate_of_fall,
        'temp_stability': thermal_stability
    })
    
    return features
