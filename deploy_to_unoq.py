"""
RetroFit 1-Click Uno Q Deployment & Mode Switcher
Automatically connects to Arduino Uno Q over Wi-Fi, uploads the latest
trained Hybrid AI model artifacts, and switches between COLLECTION and MONITORING modes.

Usage:
    python deploy_to_unoq.py --mode MONITORING   # Deploy model & start live AI monitoring
    python deploy_to_unoq.py --mode COLLECTION   # Switch back to recording healthy datasets
"""

import os
import sys
import json
import argparse
import socket
import subprocess
import paramiko

USERNAME = "arduino"
PASSWORD = "Ganesha@2003"
REMOTE_APP_DIR = "/home/arduino/ArduinoApps/retrofit/python"
KNOWN_IPS = ["10.124.6.105", "192.168.29.219", "10.51.210.105", "unoq.local", "arduino.local"]

def find_board_ip():
    for ip in KNOWN_IPS:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            res = s.connect_ex((ip, 22))
            s.close()
            if res == 0: return ip
        except Exception: pass

    try:
        output = subprocess.check_output("arp -a", shell=True).decode('utf-8', errors='ignore')
        for line in output.splitlines():
            if "14-b5-cd-e7-69-83" in line.lower() or "14-b5-cd" in line.lower():
                parts = line.split()
                if len(parts) >= 1: return parts[0]
    except Exception: pass

    return "10.124.6.105"

def deploy(mode="MONITORING", machine_id="laptop_01", session_id="idle_01"):
    workspace = os.path.dirname(os.path.abspath(__file__))
    local_models_dir = os.path.join(workspace, "models")
    local_main_py = os.path.join(workspace, "edge_main.py")
        
    board_ip = find_board_ip()
    print("=" * 65)
    print(f"  CONNECTING TO ARDUINO UNO Q @ {board_ip}")
    print(f"  TARGET COMMAND/MODE: [{mode.upper()}]")
    print("=" * 65)
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(board_ip, username=USERNAME, password=PASSWORD, timeout=8)
        
        if mode.upper() in ["STOP", "OFF"]:
            print(" -> Stopping app user:retrofit on Uno Q...")
            _, out, _ = ssh.exec_command("arduino-app-cli app stop user:retrofit")
            out.read()
            ssh.close()
            print("[SUCCESS] Data collection / monitoring stopped on Uno Q!")
            print("You can now run: python download_dataset.py")
            print("=" * 65)
            return

        sftp = ssh.open_sftp()
        
        # 1. Update main.py on the board
        if os.path.exists(local_main_py):
            print(" -> Uploading latest dual-mode engine (edge_main.py) to Uno Q...")
            sftp.put(local_main_py, f"{REMOTE_APP_DIR}/main.py")
            
        # 2. Upload Model Artifacts (if MONITORING mode)
        if mode.upper() == "MONITORING" and os.path.exists(local_models_dir):
            remote_models = f"{REMOTE_APP_DIR}/models"
            try:
                sftp.mkdir(remote_models)
            except IOError:
                pass
                
            model_files = ["scaler.pkl", "gmm_state_model.pkl", "hybrid_parameters.json", "retrofit_autoencoder.tflite", "retrofit_encoder.tflite"]
            print(f" -> Uploading trained Hybrid ML model artifacts to {remote_models}...")
            for mf in model_files:
                local_file = os.path.join(local_models_dir, mf)
                if os.path.exists(local_file):
                    sftp.put(local_file, f"{remote_models}/{mf}")
                    print(f"    - Uploaded {mf}")
                    
        # 3. Update Remote config.json
        remote_config = {
            "mode": mode.upper(),
            "machine_id": machine_id,
            "session_id": session_id,
            "mqtt_enabled": True,
            "mqtt_broker": "broker.emqx.io",
            "mqtt_port": 1883,
            "mqtt_topic": f"retrofit/telemetry/{machine_id}",
            "twilio_enabled": False
        }
        
        with sftp.file(f"{REMOTE_APP_DIR}/config.json", "w") as f:
            f.write(json.dumps(remote_config, indent=2))
        print(f" -> Updated {REMOTE_APP_DIR}/config.json to mode [{mode.upper()}]")
        
        sftp.close()

        # 4. Restart app to apply new mode
        print(" -> Restarting app user:retrofit on Uno Q...")
        _, out, _ = ssh.exec_command("arduino-app-cli app stop user:retrofit; sleep 1; arduino-app-cli app start user:retrofit")
        out.read()

        ssh.close()
        
        print("\n" + "=" * 65)
        print(f"[SUCCESS] Arduino Uno Q is now operating in [{mode.upper()}] mode!")
        if mode.upper() == "MONITORING":
            print("  - Real-time Edge AI Inference is ACTIVE")
            print("  - Live telemetry is streaming over MQTT to: retrofit/telemetry/" + machine_id)
        else:
            print(f"  - Data Acquisition is ACTIVE for machine: [{machine_id}]")
            print(f"  - Recording raw sensor data to session: [{session_id}]")
            print("  - When finished, stop it with: python deploy_to_unoq.py --mode STOP")
        print("=" * 65)
        
    except Exception as e:
        print(f"\n[ERROR] Deployment failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="MONITORING", choices=["COLLECTION", "MONITORING", "STOP", "OFF", "collection", "monitoring", "stop", "off"])
    parser.add_argument("--machine", default="laptop_01")
    parser.add_argument("--session", default="idle_01")
    args = parser.parse_args()
    deploy(mode=args.mode, machine_id=args.machine, session_id=args.session)
