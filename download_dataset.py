"""
RetroFit Smart Dataset Downloader
Automatically discovers the Arduino Uno Q on your Wi-Fi network and downloads
all recorded sensor datasets to your PC.
"""

import paramiko
import os
import sys
import socket
import subprocess
import re

import concurrent.futures
import argparse

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

    return "10.51.210.105"

def fetch_datasets(override_ip=None):
    board_ip = find_board_ip(override_ip)
    print(f"Connecting to Arduino Uno Q @ {board_ip}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(board_ip, username=USERNAME, password=PASSWORD, timeout=10)
        sftp = ssh.open_sftp()
        
        try:
            items = sftp.listdir(REMOTE_DATA_DIR)
        except IOError:
            print(f"No recordings found yet in {REMOTE_DATA_DIR}.")
            sftp.close()
            ssh.close()
            return
            
        print(f"Found {len(items)} session items in {REMOTE_DATA_DIR}\n")
        
        for item in sorted(items):
            remote_path = f"{REMOTE_DATA_DIR}/{item}"
            local_path = os.path.join(LOCAL_DEST, item)
            
            try:
                sub_items = sftp.listdir(remote_path)
                os.makedirs(local_path, exist_ok=True)
                for sub_file in sub_items:
                    r_sub = f"{remote_path}/{sub_file}"
                    l_sub = os.path.join(local_path, sub_file)
                    
                    # Only download if file does not exist locally or is different size
                    r_stat = sftp.stat(r_sub)
                    if not os.path.exists(l_sub) or os.path.getsize(l_sub) != r_stat.st_size:
                        print(f"Downloading [{item}] {sub_file} ({r_stat.st_size / (1024*1024):.2f} MB)...")
                        sftp.get(r_sub, l_sub)
                    else:
                        print(f"Already up-to-date: [{item}] {sub_file}")
            except IOError:
                r_stat = sftp.stat(remote_path)
                if not os.path.exists(local_path) or os.path.getsize(local_path) != r_stat.st_size:
                    print(f"Downloading {item}...")
                    sftp.get(remote_path, local_path)
                
        print("\n[SUCCESS] All datasets synchronized successfully to your PC!")
        sftp.close()
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Failed to fetch datasets: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=None, help="Directly specify Arduino Uno Q IP address")
    args = parser.parse_args()
    fetch_datasets(override_ip=args.ip)
