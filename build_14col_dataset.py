"""
RetroFit Dataset Transformer (Pure Python - No External Dependencies)
Merges multi-rate raw sensor streams (IMU, Audio, Temperature) into the
standard 14-column CSV format required for Martin's ML Pipeline.

Usage:
    python build_14col_dataset.py [optional_session_folder]
"""

import os
import sys
import glob
import csv
import bisect

def process_session(session_dir, output_file=None):
    if not os.path.exists(session_dir):
        print(f"Session directory not found: {session_dir}")
        return None
        
    imu_path = os.path.join(session_dir, "imu.csv")
    audio_path = os.path.join(session_dir, "audio.csv")
    temp_path = os.path.join(session_dir, "temperature.csv")
    
    if not (os.path.exists(imu_path) and os.path.exists(audio_path) and os.path.exists(temp_path)):
        print(f"Missing required CSV files in {session_dir}")
        return None

    print(f"Processing session: {session_dir}")
    
    # 1. Read Temperature Data
    temp_records = [] # (timestamp_ms, obj_temp, amb_temp)
    with open(temp_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 3:
                try:
                    temp_records.append((int(row[0]), float(row[1]), float(row[2])))
                except ValueError:
                    continue
                    
    temp_records.sort(key=lambda x: x[0])
    temp_ts = [t[0] for t in temp_records]
    
    # Helper to find closest temperature
    def get_closest_temp(ts_ms):
        if not temp_records:
            return 25.0, 25.0
        idx = bisect.bisect_left(temp_ts, ts_ms)
        if idx == 0:
            return temp_records[0][2], temp_records[0][1] # amb, obj
        if idx >= len(temp_records):
            return temp_records[-1][2], temp_records[-1][1]
        # Choose closest between idx-1 and idx
        if abs(temp_records[idx-1][0] - ts_ms) <= abs(temp_records[idx][0] - ts_ms):
            return temp_records[idx-1][2], temp_records[idx-1][1]
        else:
            return temp_records[idx][2], temp_records[idx][1]

    # 2. Read Audio Data
    audio_records = [] # (timestamp_us, adc_val)
    with open(audio_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    audio_records.append((int(row[0]), int(row[1])))
                except ValueError:
                    continue
                    
    audio_records.sort(key=lambda x: x[0])
    audio_ts = [a[0] for a in audio_records]
    
    # 3. Read IMU Data & Align
    if output_file is None:
        output_file = os.path.join(session_dir, "unified_dataset_14col.csv")
        
    out_rows = 0
    with open(imu_path, 'r', encoding='utf-8') as f_in, open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # Header: timestamp_ms,ir_ambient_c,ir_object_c,sound_peak,sound_volts,ax,ay,az,gx,gy,gz,mx,my,mz
        writer.writerow(["timestamp_ms", "ir_ambient_c", "ir_object_c", "sound_peak", "sound_volts", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"])
        
        header = next(reader, None)
        prev_imu_us = None
        
        for row in reader:
            if len(row) < 7:
                continue
            try:
                imu_us = int(row[0])
                ax, ay, az = int(row[1]), int(row[2]), int(row[3])
                gx, gy, gz = int(row[4]), int(row[5]), int(row[6])
            except ValueError:
                continue
                
            imu_ms = imu_us // 1000
            amb_temp, obj_temp = get_closest_temp(imu_ms)
            
            # Find audio samples in interval [prev_imu_us, imu_us]
            t_start = prev_imu_us if prev_imu_us is not None else (imu_us - 10000)
            t_end = imu_us
            prev_imu_us = imu_us
            
            idx_start = bisect.bisect_left(audio_ts, t_start)
            idx_end = bisect.bisect_right(audio_ts, t_end)
            
            if idx_end > idx_start:
                samples = [audio_records[k][1] for k in range(idx_start, idx_end)]
                sound_peak = max(samples) - min(samples)
                sound_volts = round((sound_peak * 3.3) / 16383.0, 3) # 14-bit ADC
            else:
                sound_peak = 0
                sound_volts = 0.0
                
            writer.writerow([
                imu_ms,
                round(amb_temp, 2),
                round(obj_temp, 2),
                sound_peak,
                sound_volts,
                ax, ay, az,
                gx, gy, gz,
                0, 0, 0
            ])
            out_rows += 1
            
    # Also write a copy to the root workspace for easy sending to Martin
    workspace_copy = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"mlx90614_dataset_converted.csv")
    try:
        import shutil
        shutil.copyfile(output_file, workspace_copy)
    except Exception:
        pass

    print(f"\n[SUCCESS] Successfully generated 14-column dataset:")
    print(f"  -> {output_file} ({out_rows} rows)")
    print(f"  -> {workspace_copy}")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) > 1:
        session_path = sys.argv[1]
    else:
        # Find all session directories and sort by mtime, picking the latest with real data
        all_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and ('laptop_' in d or 'idle_' in d)]
        valid_dirs = []
        for d in all_dirs:
            imu = os.path.join(d, 'imu.csv')
            if os.path.exists(imu) and os.path.getsize(imu) > 100:
                valid_dirs.append(d)
        if not valid_dirs:
            print("No populated session directories found.")
            sys.exit(1)
        valid_dirs.sort(key=lambda d: os.path.getmtime(d))
        session_path = valid_dirs[-1]
        
    process_session(session_path)
