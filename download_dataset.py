"""
RetroFit Smart Dataset Downloader
Automatically discovers the Arduino Uno Q, downloads ONLY the latest recorded
session (or specified session), validates data integrity (verifies row count > 0),
and automatically builds the unified 14-column dataset.
"""

import paramiko
import os
import sys
import socket
import subprocess
import concurrent.futures
import argparse
import time

USERNAME = "arduino"
PASSWORD = "Ganesha@2003"
REMOTE_DATA_DIR = "/home/arduino/ArduinoApps/retrofit/python/data"
LOCAL_DEST = os.path.dirname(os.path.abspath(__file__))

KNOWN_IPS = ["192.168.29.219", "10.51.210.105", "10.124.6.105", "unoq.local", "arduino.local"]

def find_board_ip(override_ip=None):
    if override_ip:
        return override_ip
        
    for ip in KNOWN_IPS:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.4)
            res = s.connect_ex((ip, 22))
            s.close()
            if res == 0:
                return ip
        except Exception:
            pass

    try:
        output = subprocess.check_output("arp -a", shell=True).decode('utf-8', errors='ignore')
        for line in output.splitlines():
            if "14-b5-cd-e7-69-83" in line.lower() or "14-b5-cd" in line.lower():
                parts = line.split()
                if len(parts) >= 1:
                    return parts[0]
    except Exception:
        pass

    try:
        local_ip = socket.gethostbyname(socket.gethostname())
        if local_ip.startswith("10.") or local_ip.startswith("192.168."):
            prefix = ".".join(local_ip.split(".")[:3])
            def check_subnet(i):
                tgt = f"{prefix}.{i}"
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(0.25)
                    if s.connect_ex((tgt, 22)) == 0:
                        s.close()
                        return tgt
                    s.close()
                except Exception: pass
                return None
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
                for r in ex.map(check_subnet, range(1, 255)):
                    if r: return r
    except Exception: pass

    return "192.168.29.219"

def count_csv_rows(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = sum(1 for line in f if line.strip())
    return max(0, lines - 1)

def fetch_datasets(override_ip=None, target_session=None, fetch_all=False):
    board_ip = find_board_ip(override_ip)
    print("=" * 65)
    print(f"  CONNECTING TO ARDUINO UNO Q @ {board_ip}")
    print("=" * 65)
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(board_ip, username=USERNAME, password=PASSWORD, timeout=8)
        sftp = ssh.open_sftp()
        
        try:
            items = sftp.listdir(REMOTE_DATA_DIR)
        except IOError:
            print(f"[ERROR] Remote data directory {REMOTE_DATA_DIR} not found.")
            sftp.close()
            ssh.close()
            return None

        session_dirs = []
        for item in items:
            remote_path = f"{REMOTE_DATA_DIR}/{item}"
            try:
                attr = sftp.stat(remote_path)
                if attr.st_mode & 0o040000:
                    session_dirs.append((item, attr.st_mtime))
            except Exception:
                pass

        if not session_dirs:
            print(f"[WARNING] No recorded session folders found in {REMOTE_DATA_DIR}.")
            sftp.close()
            ssh.close()
            return None

        session_dirs.sort(key=lambda x: x[1], reverse=True)

        if target_session:
            selected_sessions = [d[0] for d in session_dirs if target_session.lower() in d[0].lower()]
            if not selected_sessions:
                print(f"[ERROR] Session '{target_session}' not found on Uno Q.")
                print(f"Available sessions: {[d[0] for d in session_dirs]}")
                sftp.close()
                ssh.close()
                return None
        elif fetch_all:
            selected_sessions = [d[0] for d in session_dirs]
        else:
            selected_sessions = [session_dirs[0][0]]

        print(f" -> Selected Session to download: {selected_sessions}\n")

        latest_downloaded_dir = None
        for session in selected_sessions:
            remote_path = f"{REMOTE_DATA_DIR}/{session}"
            local_path = os.path.join(LOCAL_DEST, session)
            os.makedirs(local_path, exist_ok=True)
            
            print(f"Downloading session: [{session}]...")
            try:
                files = sftp.listdir(remote_path)
                for fname in files:
                    r_file = f"{remote_path}/{fname}"
                    l_file = os.path.join(local_path, fname)
                    sftp.get(r_file, l_file)
                    size_kb = os.path.getsize(l_file) / 1024.0
                    print(f"  - {fname:16s} ({size_kb:8.1f} KB)")
                latest_downloaded_dir = local_path
            except Exception as e:
                print(f"  [ERROR] Failed to download {session}: {e}")

        sftp.close()
        ssh.close()

        if latest_downloaded_dir and os.path.exists(latest_downloaded_dir):
            imu_file = os.path.join(latest_downloaded_dir, "imu.csv")
            aud_file = os.path.join(latest_downloaded_dir, "audio.csv")
            tmp_file = os.path.join(latest_downloaded_dir, "temperature.csv")

            imu_rows = count_csv_rows(imu_file)
            aud_rows = count_csv_rows(aud_file)
            tmp_rows = count_csv_rows(tmp_file)

            print("\n" + "=" * 65)
            print(f"  DATA INTEGRITY VERIFICATION: [{os.path.basename(latest_downloaded_dir)}]")
            print(f"  - IMU Data Rows:         {imu_rows:,} rows  ({imu_rows/50.0:.1f}s @ 50 Hz)")
            print(f"  - Audio Samples:         {aud_rows:,} samples ({aud_rows/500.0:.1f}s @ 500 Hz)")
            print(f"  - Temperature Readings:  {tmp_rows:,} readings")
            print("=" * 65)

            if imu_rows == 0:
                print("\n[CRITICAL WARNING] imu.csv has 0 data rows!")
                print("The recording session was stopped before sensor data was received.")
                return None
            else:
                print("\n -> Building unified 14-column dataset from downloaded session...")
                try:
                    import build_14col_dataset
                    build_14col_dataset.process_session(latest_downloaded_dir)
                except Exception as e:
                    print(f"[WARNING] Auto-builder notice: {e}")

        print("\n[SUCCESS] Download & dataset preparation complete!")
        return latest_downloaded_dir

    except Exception as e:
        print(f"[ERROR] Failed to connect / download: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download recorded sensor datasets from Arduino Uno Q.")
    parser.add_argument("--ip", default=None, help="Directly specify Arduino Uno Q IP address")
    parser.add_argument("--session", default=None, help="Download a specific session name (default: latest)")
    parser.add_argument("--all", action="store_true", help="Download all historical sessions (default: latest only)")
    args = parser.parse_args()
    
    fetch_datasets(override_ip=args.ip, target_session=args.session, fetch_all=args.all)
