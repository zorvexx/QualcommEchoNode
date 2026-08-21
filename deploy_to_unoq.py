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

import concurrent.futures

USERNAME = "arduino"
PASSWORD = "Ganesha@2003"
REMOTE_APP_DIR = "/home/arduino/ArduinoApps/retrofit/python"
KNOWN_IPS = ["10.51.210.105", "10.124.6.105", "192.168.29.219", "unoq.local", "arduino.local"]

def find_board_ip(override_ip=None):
    if override_ip:
        return override_ip
        
    for ip in KNOWN_IPS:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.4)
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

    # Sweep local /24 subnet if on Wi-Fi (e.g. 10.51.210.X)
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

def deploy(mode="MONITORING", machine_id="laptop_01", session_id="idle_01", override_ip=None):
    workspace = os.path.dirname(os.path.abspath(__file__))
    local_models_dir = os.path.join(workspace, "models")
    local_main_py = os.path.join(workspace, "edge_main.py")
        
    board_ip = find_board_ip(override_ip)
    print("=" * 65)
    print(f"  CONNECTING TO ARDUINO UNO Q @ {board_ip}")
    print(f"  TARGET COMMAND/MODE: [{mode.upper()}]")
    print("=" * 65)
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(board_ip, username=USERNAME, password=PASSWORD, timeout=8)

        # ── SERIAL_COLLECTION: bypass broken Bridge container completely ──
        if mode.upper() == "SERIAL_COLLECTION":
            sftp = ssh.open_sftp()
            print(" -> Uploading sketch_serial.ino (plain Serial, no Bridge)...")
            local_serial_sketch = os.path.join(workspace, "sketch_serial.ino")
            if not os.path.exists(local_serial_sketch):
                print("[ERROR] sketch_serial.ino not found in project folder!")
                ssh.close(); return
            sftp.put(local_serial_sketch, "/home/arduino/ArduinoApps/retrofit/sketch/sketch.ino")

            print(" -> Uploading collect_serial.py to /tmp/collect.py on Uno Q...")
            local_collector = os.path.join(workspace, "collect_serial.py")
            if not os.path.exists(local_collector):
                print("[ERROR] collect_serial.py not found!")
                ssh.close(); return
            sftp.put(local_collector, "/tmp/collect.py")
            sftp.close()

            # Stop existing app first
            print(" -> Stopping existing RetroFit app...")
            _, out, _ = ssh.exec_command("arduino-app-cli app stop user:retrofit 2>/dev/null; sleep 1")
            out.read()

            # Flash the new sketch via arduino-app-cli (compiles and flashes STM32)
            print(" -> Compiling & flashing sketch_serial.ino to STM32 (takes ~30s)...")
            _, out, err = ssh.exec_command(
                "arduino-app-cli app start user:retrofit 2>&1 | tail -5"
            )
            import time; time.sleep(35)
            print(out.read().decode("utf-8", errors="replace")[:500])

            # Stop the app so it releases ttyHS1, then run our collector
            _, out, _ = ssh.exec_command("arduino-app-cli app stop user:retrofit 2>/dev/null; sleep 2")
            out.read()

            # Kill any previous collect.py
            ssh.exec_command("pkill -f 'collect.py' 2>/dev/null")
            time.sleep(1)

            # Install pyserial if needed
            ssh.exec_command("pip3 install pyserial --quiet 2>/dev/null &")
            time.sleep(3)

            # Start collector in background (nohup so it survives ssh disconnect)
            cmd = f"nohup python3 /tmp/collect.py {machine_id} {session_id} > /tmp/collect_log.txt 2>&1 &"
            print(f" -> Starting host-side serial collector ({machine_id}/{session_id})...")
            ssh.exec_command(cmd)
            time.sleep(3)

            # Verify it's running
            _, out, _ = ssh.exec_command("ps aux | grep collect.py | grep -v grep")
            running = out.read().decode("utf-8", errors="replace").strip()
            ssh.close()

            print("\n" + "=" * 65)
            if running:
                print(f"[SUCCESS] Serial collector is RUNNING on Uno Q!")
                print(f"  - Machine: [{machine_id}]  Session: [{session_id}]")
                print(f"  - Live log: ssh arduino@{board_ip} 'tail -f /tmp/collect_log.txt'")
                print(f"  - When done, stop with: python deploy_to_unoq.py --mode STOP")
            else:
                print("[WARNING] Collector may not have started. Check:")
                print(f"  ssh arduino@{board_ip} 'cat /tmp/collect_log.txt'")
            print("=" * 65)
            return

        if mode.upper() in ["STOP", "OFF"]:
            print(" -> Stopping app user:retrofit on Uno Q...")
            ssh.exec_command("pkill -f 'collect.py' 2>/dev/null")
            _, out, _ = ssh.exec_command("arduino-app-cli app stop user:retrofit")
            out.read()
            ssh.close()
            print("[SUCCESS] Data collection / monitoring stopped on Uno Q!")
            print("You can now run: python download_dataset.py")
            print("=" * 65)
            return

        sftp = ssh.open_sftp()
        
        # 1. Update sketch.ino on the board
        local_sketch = os.path.join(workspace, "sketch.ino")
        if os.path.exists(local_sketch):
            print(" -> Uploading latest STM32 firmware (sketch.ino) to Uno Q...")
            sftp.put(local_sketch, "/home/arduino/ArduinoApps/retrofit/sketch/sketch.ino")

        # 2. Update main.py on the board
        if os.path.exists(local_main_py):
            print(" -> Uploading latest dual-mode engine (edge_main.py) to Uno Q...")
            sftp.put(local_main_py, f"{REMOTE_APP_DIR}/main.py")
            
        # 3. Upload Model Artifacts (if MONITORING mode)
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
                    
        # 4. Update Remote config.json
        twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        env_path = os.path.join(workspace, ".env")
        if os.path.exists(env_path):
            with open(env_path) as ef:
                for line in ef:
                    if "TWILIO_ACCOUNT_SID=" in line: twilio_sid = line.split("=", 1)[1].strip()
                    elif "TWILIO_AUTH_TOKEN=" in line: twilio_token = line.split("=", 1)[1].strip()
                    
        remote_config = {
            "mode": mode.upper(),
            "machine_id": machine_id,
            "session_id": session_id,
            "mqtt_enabled": True,
            "mqtt_broker": "broker.emqx.io",
            "mqtt_port": 1883,
            "mqtt_topic": f"retrofit/telemetry/{machine_id}",
            "twilio": {
                "enabled": bool(twilio_sid and twilio_token and "YOUR" not in twilio_sid),
                "account_sid": twilio_sid,
                "auth_token": twilio_token,
                "from_phone": os.getenv("TWILIO_FROM_PHONE", "+15708730348"),
                "target_phone": os.getenv("TWILIO_TARGET_PHONE", "+918401782327"),
                "cooldown_seconds": 60
            }
        }
        
        with sftp.file(f"{REMOTE_APP_DIR}/config.json", "w") as f:
            f.write(json.dumps(remote_config, indent=2))
        print(f" -> Updated {REMOTE_APP_DIR}/config.json to mode [{mode.upper()}] (Twilio: {'ACTIVE' if remote_config['twilio']['enabled'] else 'DISABLED'})")
        
        sftp.close()

        # 5. Compile firmware, flash MCU, and start engine
        print(" -> Compiling firmware and starting app on Uno Q (takes ~20s)...")
        _, out, _ = ssh.exec_command("arduino-app-cli app stop user:retrofit 2>/dev/null; sleep 1; arduino-app-cli app start user:retrofit 2>&1")
        out.read()
        import time
        time.sleep(18)

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
    parser.add_argument("--mode", default="MONITORING", choices=["COLLECTION", "SERIAL_COLLECTION", "MONITORING", "STOP", "OFF", "collection", "serial_collection", "monitoring", "stop", "off"])
    parser.add_argument("--machine", default="laptop_01")
    parser.add_argument("--session", default="idle_01")
    parser.add_argument("--ip", default=None, help="Directly specify Arduino Uno Q IP address")
    args = parser.parse_args()
    deploy(mode=args.mode, machine_id=args.machine, session_id=args.session, override_ip=args.ip)
