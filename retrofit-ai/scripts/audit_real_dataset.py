import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_real_dataset_audit(csv_path=r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv", output_dir="data/reports/real_data_audit"):
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"[AUDIT] Reading real sensor dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Samples and Duration
    num_samples = len(df)
    
    # Check timestamp column
    ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else df.columns[0]
    ts = df[ts_col].values.astype(float)
    
    # Duration
    duration_ms = ts[-1] - ts[0]
    duration_sec = duration_ms / 1000.0
    
    # Timestamp differences (dt)
    dt_ms = np.diff(ts)
    dt_sec = dt_ms / 1000.0
    
    # Monotonicity check
    is_monotonic = bool(np.all(dt_ms >= 0))
    strict_monotonic = bool(np.all(dt_ms > 0))
    
    # Duplicates
    num_duplicates = np.sum(dt_ms == 0)
    
    # Negative time jumps
    num_negative = np.sum(dt_ms < 0)
    
    # Effective sampling frequency
    eff_fs_hz = (num_samples - 1) / duration_sec if duration_sec > 0 else 0.0
    median_dt_ms = np.median(dt_ms) if len(dt_ms) > 0 else 0.0
    median_fs_hz = 1000.0 / median_dt_ms if median_dt_ms > 0 else 0.0
    
    # Gaps
    gaps_gt_10ms = np.sum(dt_ms > 10.0)
    gaps_gt_20ms = np.sum(dt_ms > 20.0)
    gaps_gt_50ms = np.sum(dt_ms > 50.0)
    gaps_gt_100ms = np.sum(dt_ms > 100.0)
    
    # Sampling frequency distribution
    dt_valid = dt_ms[dt_ms > 0]
    fs_inst = 1000.0 / dt_valid if len(dt_valid) > 0 else np.array([0.0])
    
    # Sensor Magnitudes
    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
    
    # Modalities to inspect (excluding mx, my, mz)
    active_cols = ['ax', 'ay', 'az', 'acc_mag', 'gx', 'gy', 'gz', 'gyro_mag', 'sound_peak', 'sound_volts', 'ir_ambient_c', 'ir_object_c']
    
    sensor_stats = []
    for col in active_cols:
        if col in df.columns:
            vals = df[col].dropna().values
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            num_outliers = np.sum((vals < lower_bound) | (vals > upper_bound))
            
            sensor_stats.append({
                'channel': col,
                'min': round(float(np.min(vals)), 4),
                'max': round(float(np.max(vals)), 4),
                'mean': round(float(np.mean(vals)), 4),
                'std': round(float(np.std(vals)), 4),
                'rms': round(float(np.sqrt(np.mean(vals**2))), 4),
                'peak_to_peak': round(float(np.max(vals) - np.min(vals)), 4),
                'outliers_count': int(num_outliers),
                'is_constant': bool(np.std(vals) == 0)
            })
            
    df_sensor_stats = pd.DataFrame(sensor_stats)
    df_sensor_stats.to_csv(os.path.join(output_dir, "sensor_statistics.csv"), index=False)
    
    # Generate Plots
    time_sec = (ts - ts[0]) / 1000.0
    
    # 1. Accelerometer plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_sec, df['ax'], label='ax', alpha=0.8)
    plt.plot(time_sec, df['ay'], label='ay', alpha=0.8)
    plt.plot(time_sec, df['az'], label='az', alpha=0.8)
    plt.plot(time_sec, df['acc_mag'], label='acc_mag', color='black', linewidth=1.5)
    plt.title("Real MPU6050 Accelerometer & Magnitude Over Time (Laptop Idle)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration (g or m/s²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "01_accelerometer_time_series.png"))
    plt.close()
    
    # 2. Gyroscope plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_sec, df['gx'], label='gx', alpha=0.8)
    plt.plot(time_sec, df['gy'], label='gy', alpha=0.8)
    plt.plot(time_sec, df['gz'], label='gz', alpha=0.8)
    plt.plot(time_sec, df['gyro_mag'], label='gyro_mag', color='black', linewidth=1.5)
    plt.title("Real MPU6050 Gyroscope & Magnitude Over Time (Laptop Idle)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "02_gyroscope_time_series.png"))
    plt.close()
    
    # 3. Object Temperature plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_sec, df['ir_object_c'], color='crimson', linewidth=1.5, label='ir_object_c')
    plt.title("Real MLX90614 Object Temperature Over Time (°C)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "03_object_temperature_time_series.png"))
    plt.close()
    
    # 4. Ambient Temperature plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_sec, df['ir_ambient_c'], color='darkorange', linewidth=1.5, label='ir_ambient_c')
    plt.title("Real MLX90614 Ambient Temperature Over Time (°C)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "04_ambient_temperature_time_series.png"))
    plt.close()
    
    # 5. Sound Peak plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_sec, df['sound_peak'], color='purple', alpha=0.8, label='sound_peak')
    plt.title("Real MAX9814 Sound Peak Over Time (Noisy Room Ambient)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Sound Peak (Raw ADC / Amplitude)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "05_sound_peak_time_series.png"))
    plt.close()
    
    # 6. Sound Volts plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_sec, df['sound_volts'], color='teal', alpha=0.8, label='sound_volts')
    plt.title("Real MAX9814 Sound Volts Over Time (V)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Sound Voltage (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "06_sound_volts_time_series.png"))
    plt.close()
    
    # 7. Timestamp Interval Distribution Plot
    plt.figure(figsize=(10, 4))
    plt.hist(dt_ms, bins=50, color='navy', edgecolor='black', alpha=0.7)
    plt.axvline(median_dt_ms, color='red', linestyle='--', label=f'Median dt: {median_dt_ms:.2f} ms ({median_fs_hz:.1f} Hz)')
    plt.title("Timestamp Sampling Interval Distribution (Δt in ms)")
    plt.xlabel("Interval Δt (ms)")
    plt.ylabel("Frequency Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "07_timestamp_interval_distribution.png"))
    plt.close()
    
    # Output Audit Metrics Summary JSON
    audit_summary = {
        'num_samples': num_samples,
        'recording_duration_sec': round(duration_sec, 2),
        'timestamp_monotonic': is_monotonic,
        'strictly_increasing': strict_monotonic,
        'timestamp_duplicates_count': int(num_duplicates),
        'negative_time_jumps_count': int(num_negative),
        'effective_sampling_freq_hz': round(eff_fs_hz, 2),
        'median_sampling_freq_hz': round(median_fs_hz, 2),
        'median_dt_ms': round(median_dt_ms, 2),
        'dt_min_ms': round(float(np.min(dt_ms)), 2),
        'dt_max_ms': round(float(np.max(dt_ms)), 2),
        'dt_mean_ms': round(float(np.mean(dt_ms)), 2),
        'dt_std_ms': round(float(np.std(dt_ms)), 2),
        'dt_p25_ms': round(float(np.percentile(dt_ms, 25)), 2),
        'dt_p75_ms': round(float(np.percentile(dt_ms, 75)), 2),
        'dt_p95_ms': round(float(np.percentile(dt_ms, 95)), 2),
        'gaps_gt_10ms': int(gaps_gt_10ms),
        'gaps_gt_20ms': int(gaps_gt_20ms),
        'gaps_gt_50ms': int(gaps_gt_50ms),
        'gaps_gt_100ms': int(gaps_gt_100ms),
        'missing_values_total': int(df[active_cols].isna().sum().sum())
    }
    
    with open(os.path.join(output_dir, "audit_summary.json"), 'w') as f:
        json.dump(audit_summary, f, indent=2)
        
    print("\n--- REAL DATASET AUDIT COMPLETED ---")
    print(json.dumps(audit_summary, indent=2))
    print("\nSensor Statistics:")
    print(df_sensor_stats.to_string(index=False))
    return audit_summary, df_sensor_stats

if __name__ == "__main__":
    run_real_dataset_audit()
