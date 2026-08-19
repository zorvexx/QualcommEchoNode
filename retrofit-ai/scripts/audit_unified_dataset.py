import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def audit_unified_dataset(csv_path="data/raw/unified_dataset_14col.csv", output_plot_dir="data/plots"):
    print("=========================================================")
    print("      RETROFIT REAL DATASET SEGMENTATION AUDIT           ")
    print("=========================================================")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset file not found at: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    os.makedirs(output_plot_dir, exist_ok=True)
    
    sample_count = len(df)
    cols = df.columns.tolist()
    
    print(f" -> Dataset File Path  : {csv_path}")
    print(f" -> Total Sample Count : {sample_count} samples")
    print(f" -> Columns Found      : {cols}")
    
    # Check missing values
    missing_sum = df.isnull().sum().to_dict()
    total_missing = sum(missing_sum.values())
    print(f" -> Total Missing Val  : {total_missing}")
    
    # Timestamp Analysis
    ts = df['timestamp_ms'].values
    is_monotonic = bool(np.all(np.diff(ts) >= 0))
    dt = np.diff(ts)
    
    duration_sec = (ts[-1] - ts[0]) / 1000.0
    dt_mean = float(np.mean(dt))
    dt_median = float(np.median(dt))
    dt_min = float(np.min(dt))
    dt_max = float(np.max(dt))
    eff_fs = float(1000.0 / dt_mean) if dt_mean > 0 else 0.0
    
    gaps_gt_10 = int(np.sum(dt > 10.0))
    gaps_gt_20 = int(np.sum(dt > 20.0))
    gaps_gt_50 = int(np.sum(dt > 50.0))
    gaps_gt_100 = int(np.sum(dt > 100.0))
    
    print("\n--- 1. TIMESTAMP & SAMPLING PROFILE ---")
    print(f"  Total Duration      : {duration_sec:.2f} s ({duration_sec/60.0:.2f} minutes)")
    print(f"  Monotonicity        : {'STRICTLY MONOTONIC' if is_monotonic else 'NON-MONOTONIC DETECTED!'}")
    print(f"  Sampling Interval   : Mean={dt_mean:.2f}ms | Median={dt_median:.2f}ms | Min={dt_min:.2f}ms | Max={dt_max:.2f}ms")
    print(f"  Effective Fs        : {eff_fs:.2f} Hz")
    print(f"  Gaps > 10ms         : {gaps_gt_10}")
    print(f"  Gaps > 20ms         : {gaps_gt_20}")
    print(f"  Gaps > 50ms         : {gaps_gt_50}")
    print(f"  Gaps > 100ms        : {gaps_gt_100}")
    
    # Accelerometer Magnitude (Exclude Magnetometer)
    ax, ay, az = df['ax'].values, df['ay'].values, df['az'].values
    amag = np.sqrt(ax**2 + ay**2 + az**2)
    df['amag'] = amag
    
    # Gyro Magnitude
    gx, gy, gz = df['gx'].values, df['gy'].values, df['gz'].values
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)
    df['gmag'] = gmag
    
    # Time vector in seconds
    t_sec = (ts - ts[0]) / 1000.0
    df['t_sec'] = t_sec
    
    # Rolling Statistics (200 samples window ~ 2-3 seconds)
    roll_win = min(200, sample_count // 5)
    df['amag_roll_var'] = df['amag'].rolling(roll_win, min_periods=10).var()
    df['gmag_roll_var'] = df['gmag'].rolling(roll_win, min_periods=10).var()
    df['ir_obj_roll_mean'] = df['ir_object_c'].rolling(roll_win, min_periods=10).mean()
    df['sound_volts_roll_mean'] = df['sound_volts'].rolling(roll_win, min_periods=10).mean()
    
    # Print Sensor Ranges
    print("\n--- 2. SENSOR SIGNAL RANGES ---")
    sensor_cols = ['ir_ambient_c', 'ir_object_c', 'sound_peak', 'sound_volts', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    range_summary = {}
    for sc in sensor_cols:
        if sc in df.columns:
            s_data = df[sc].values
            range_summary[sc] = {
                'min': round(float(np.min(s_data)), 2),
                'max': round(float(np.max(s_data)), 2),
                'mean': round(float(np.mean(s_data)), 2),
                'std': round(float(np.std(s_data)), 2)
            }
            print(f"  {sc:15s}: Min={range_summary[sc]['min']:8.2f} | Max={range_summary[sc]['max']:8.2f} | Mean={range_summary[sc]['mean']:8.2f} | Std={range_summary[sc]['std']:8.2f}")

    # Temporal Segmentation & Change-Point Analysis
    # Let's inspect rolling variance to find stable vs transition vs perturbation regions
    roll_v = df['amag_roll_var'].fillna(0).values
    high_var_thresh = np.percentile(roll_v, 80)
    
    print("\n--- 3. TEMPORAL SEGMENTATION & REGION IDENTIFICATION ---")
    # Identify continuous baseline vs perturbation chunks
    is_stable = roll_v < high_var_thresh
    
    # Calculate stable region statistics
    stable_sec = float(np.sum(is_stable) * dt_mean / 1000.0)
    print(f"  Estimated Stable / Baseline Duration : {stable_sec:.2f} s ({stable_sec/duration_sec*100:.1f}% of recording)")
    print(f"  High Dynamic / Perturbation Duration : {duration_sec - stable_sec:.2f} s ({(1-stable_sec/duration_sec)*100:.1f}% of recording)")

    # -------------------------------------------------------------
    # GENERATE PLOTS
    # -------------------------------------------------------------
    plt.style.use('dark_background')
    
    # Plot 1: Temperature vs Time
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t_sec, df['ir_ambient_c'], label='Ambient Temp (°C)', color='#38bdf8', alpha=0.8)
    ax1.plot(t_sec, df['ir_object_c'], label='Object Temp (°C)', color='#f43f5e', linewidth=1.5)
    ax1.set_title("RetroFit Real Sensor Data — Temperature vs Time")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left')
    p1 = os.path.join(output_plot_dir, "temperature_vs_time.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()
    
    # Plot 2: Acceleration Magnitude vs Time
    fig, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(t_sec, amag, label='Accel Magnitude (LSB)', color='#4ade80', alpha=0.7, linewidth=0.8)
    ax2.set_title("RetroFit Real Sensor Data — Acceleration Magnitude vs Time")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Acceleration Magnitude (LSB)")
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper left')
    p2 = os.path.join(output_plot_dir, "acc_mag_vs_time.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()

    # Plot 3: Gyro Magnitude vs Time
    fig, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(t_sec, gmag, label='Gyro Magnitude (LSB)', color='#fbbf24', alpha=0.7, linewidth=0.8)
    ax3.set_title("RetroFit Real Sensor Data — Gyro Magnitude vs Time")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Gyro Magnitude (LSB)")
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(loc='upper left')
    p3 = os.path.join(output_plot_dir, "gyro_mag_vs_time.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()

    # Plot 4: Acoustic Amplitude vs Time
    fig, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(t_sec, df['sound_volts'], label='Sound Volts (V)', color='#a78bfa', alpha=0.8)
    ax4.plot(t_sec, df['sound_peak'], label='Sound Peak (ADC)', color='#f472b6', alpha=0.5)
    ax4.set_title("RetroFit Real Sensor Data — MAX9814 Acoustic Amplitude vs Time")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Voltage / Amplitude")
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.legend(loc='upper left')
    p4 = os.path.join(output_plot_dir, "acoustic_vs_time.png")
    plt.tight_layout()
    plt.savefig(p4, dpi=150)
    plt.close()

    # Plot 5: Rolling Variance
    fig, ax5 = plt.subplots(figsize=(10, 5))
    ax5.plot(t_sec, df['amag_roll_var'], label='Accel Rolling Var', color='#4ade80')
    ax5.plot(t_sec, df['gmag_roll_var'], label='Gyro Rolling Var', color='#fbbf24', alpha=0.8)
    ax5.set_title("RetroFit Real Sensor Data — Rolling Variance (Change-Point Analysis)")
    ax5.set_xlabel("Time (seconds)")
    ax5.set_ylabel("Variance")
    ax5.grid(True, linestyle='--', alpha=0.3)
    ax5.legend(loc='upper left')
    p5 = os.path.join(output_plot_dir, "rolling_variance.png")
    plt.tight_layout()
    plt.savefig(p5, dpi=150)
    plt.close()

    # Plot 6: Timestamp Gaps
    fig, ax6 = plt.subplots(figsize=(10, 5))
    ax6.plot(t_sec[1:], dt, label='$\Delta t$ Interval (ms)', color='#f87171', alpha=0.7, linewidth=0.8)
    ax6.axhline(dt_mean, color='#38bdf8', linestyle='--', label=f'Mean dt ({dt_mean:.1f}ms)')
    ax6.set_title("RetroFit Real Sensor Data — Sampling Timestamp Gaps")
    ax6.set_xlabel("Time (seconds)")
    ax6.set_ylabel("Interval $\Delta t$ (ms)")
    ax6.grid(True, linestyle='--', alpha=0.3)
    ax6.legend(loc='upper left')
    p6 = os.path.join(output_plot_dir, "timestamp_gaps.png")
    plt.tight_layout()
    plt.savefig(p6, dpi=150)
    plt.close()
    
    print("\n--- 4. GENERATED DIAGNOSTIC PLOTS ---")
    print(f"  1. Temperature vs Time  -> {p1}")
    print(f"  2. Accel Mag vs Time    -> {p2}")
    print(f"  3. Gyro Mag vs Time     -> {p3}")
    print(f"  4. Acoustic vs Time     -> {p4}")
    print(f"  5. Rolling Variance     -> {p5}")
    print(f"  6. Timestamp Gaps       -> {p6}")

    summary = {
        'duration_sec': round(duration_sec, 2),
        'sample_count': sample_count,
        'is_monotonic': is_monotonic,
        'dt_mean': round(dt_mean, 2),
        'dt_median': round(dt_median, 2),
        'dt_min': round(dt_min, 2),
        'dt_max': round(dt_max, 2),
        'eff_fs': round(eff_fs, 2),
        'gaps_gt_10': gaps_gt_10,
        'gaps_gt_20': gaps_gt_20,
        'gaps_gt_50': gaps_gt_50,
        'gaps_gt_100': gaps_gt_100,
        'total_missing': total_missing,
        'stable_sec': round(stable_sec, 2),
        'range_summary': range_summary
    }
    
    with open("data/unified_dataset_audit.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary

if __name__ == "__main__":
    audit_unified_dataset()
