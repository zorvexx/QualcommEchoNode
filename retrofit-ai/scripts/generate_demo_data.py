import os
import argparse
import numpy as np
import pandas as pd

def generate_demo_dataset(output_path="data/raw/demo_machine.csv", duration_seconds=600, fs_accel=1000, fs_audio=16000, fs_temp=10):
    """
    Generates synthetic realistic machine sensor data with:
    - Normal operating states (IDLE, STARTUP, NORMAL_OPERATION, HIGH_LOAD, SHUTDOWN)
    - Controlled abnormal conditions (vibration spikes, acoustic noise, temperature rise)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    dt_accel = 1.0 / fs_accel
    t_accel = np.arange(0, duration_seconds, dt_accel)
    num_samples = len(t_accel)
    
    # Base timestamps
    timestamps = 1770000000.0 + t_accel
    
    # State segments
    # 0 - 60s: STARTUP
    # 60 - 240s: NORMAL_OPERATION (Session 1)
    # 240 - 360s: HIGH_LOAD (Session 1)
    # 360 - 480s: NORMAL_OPERATION (Session 2)
    # 480 - 540s: ABNORMAL (Vibration + Acoustic + Temp rise)
    # 540 - 600s: SHUTDOWN
    
    operating_state = []
    session_id = []
    labels = []
    
    ax = np.zeros(num_samples)
    ay = np.zeros(num_samples)
    az = np.ones(num_samples) * 9.81
    
    gx = np.zeros(num_samples)
    gy = np.zeros(num_samples)
    gz = np.zeros(num_samples)
    
    mx = np.ones(num_samples) * 20.0
    my = np.ones(num_samples) * 5.0
    mz = np.ones(num_samples) * 45.0
    
    audio = np.zeros(num_samples)
    temp = np.ones(num_samples) * 25.0
    
    for i, t in enumerate(t_accel):
        # Baseline noise
        ax[i] = 0.1 * np.sin(2 * np.pi * 25 * t) + np.random.normal(0, 0.05)
        ay[i] = 0.1 * np.cos(2 * np.pi * 25 * t) + np.random.normal(0, 0.05)
        az[i] = 9.81 + 0.05 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.05)
        
        gx[i] = np.random.normal(0, 0.02)
        gy[i] = np.random.normal(0, 0.02)
        gz[i] = 0.5 * np.sin(2 * np.pi * 25 * t) + np.random.normal(0, 0.02)
        
        mx[i] = 20.0 + 0.2 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.05)
        my[i] = 5.0 + 0.2 * np.cos(2 * np.pi * 50 * t) + np.random.normal(0, 0.05)
        mz[i] = 45.0 + np.random.normal(0, 0.05)
        
        audio[i] = 0.05 * np.sin(2 * np.pi * 440 * t) + np.random.normal(0, 0.02)
        temp[i] = 25.0 + 0.01 * (t / 60.0)
        
        if t < 60:
            operating_state.append("STARTUP")
            session_id.append(1)
            labels.append(0)
            ax[i] *= 1.5
            ay[i] *= 1.5
        elif t < 240:
            operating_state.append("NORMAL_OPERATION")
            session_id.append(1)
            labels.append(0)
        elif t < 360:
            operating_state.append("HIGH_LOAD")
            session_id.append(1)
            labels.append(0)
            ax[i] *= 1.8
            ay[i] *= 1.8
            temp[i] += 5.0
            audio[i] *= 1.4
        elif t < 480:
            operating_state.append("NORMAL_OPERATION")
            session_id.append(2)
            labels.append(0)
        elif t < 540:
            # Abnormal condition (Fault!)
            operating_state.append("HIGH_LOAD")
            session_id.append(3)
            labels.append(1)
            # Severe kurtosis impact shocks & friction acoustics
            if i % 100 == 0:
                ax[i] += 4.5
                ay[i] += 4.5
            ax[i] *= 3.2
            ay[i] *= 3.2
            audio[i] += 0.3 * np.random.normal(0, 0.2)
            temp[i] += 18.0 + (t - 480) * 0.1
        else:
            operating_state.append("SHUTDOWN")
            session_id.append(3)
            labels.append(0)
            ax[i] *= 0.2
            ay[i] *= 0.2
            audio[i] *= 0.1
            
    df = pd.DataFrame({
        'timestamp': timestamps,
        'machine_id': 'RETROFIT_DEMO_01',
        'session_id': session_id,
        'operating_state': operating_state,
        'label': labels,
        'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz,
        'mx': mx, 'my': my, 'mz': mz,
        'audio': audio,
        'temperature': temp
    })
    
    df.to_csv(output_path, index=False)
    print(f"[DEMO DATA] Generated {len(df)} synthetic sensor samples -> {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/demo_machine.csv", help="Output raw CSV path")
    parser.add_argument("--duration", type=int, default=600, help="Duration in seconds")
    args = parser.parse_args()
    generate_demo_dataset(args.output, args.duration)
