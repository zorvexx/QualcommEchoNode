from arduino.app_utils import App, Bridge
import os
import sys
import json
import time
import datetime
import threading
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

for p in ["/home/arduino/.local/lib/python3.13/site-packages", "/root/.local/lib/python3.13/site-packages"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

config = {
    "mode": "MONITORING",
    "machine_id": "laptop_01",
    "session_id": "idle_01",
    "mqtt_enabled": True,
    "mqtt_broker": "broker.emqx.io",
    "mqtt_port": 1883,
    "mqtt_topic": "retrofit/telemetry/laptop_01",
    "twilio": {
        "enabled": True,
        "target_phone": os.environ.get("TWILIO_TARGET_PHONE", "+918401782327"),
        "account_sid": os.environ.get("TWILIO_ACCOUNT_SID", "AC_YOUR_TWILIO_ACCOUNT_SID_HERE"),
        "auth_token": os.environ.get("TWILIO_AUTH_TOKEN", "YOUR_TWILIO_AUTH_TOKEN_HERE"),
        "from_phone": os.environ.get("TWILIO_FROM_PHONE", "+15708730348"),
        "cooldown_seconds": 60
    }
}

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            config.update(json.load(f))
    except Exception:
        pass

MODE = config.get("mode", "MONITORING").upper()
MACHINE_ID = config.get("machine_id", "laptop_01")
SESSION_ID = config.get("session_id", "idle_01")
TWILIO_CFG = config["twilio"]

# Setup CSV writers for COLLECTION mode
imu_csv_file = None
audio_csv_file = None
temp_csv_file = None

if MODE == "COLLECTION":
    SESSION_DIR = os.path.join(DATA_DIR, f"{MACHINE_ID}_{SESSION_ID}")
    os.makedirs(SESSION_DIR, exist_ok=True)
    
    # Initialize metadata.json
    meta_path = os.path.join(SESSION_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"machine_id": MACHINE_ID, "session_id": SESSION_ID, "start_time_iso": datetime.datetime.now().isoformat()}, f, indent=2)
            
    imu_path = os.path.join(SESSION_DIR, "imu.csv")
    write_imu_hdr = not os.path.exists(imu_path) or os.path.getsize(imu_path) == 0
    imu_csv_file = open(imu_path, "a", encoding="utf-8")
    if write_imu_hdr:
        imu_csv_file.write("timestamp_us,ax,ay,az,gx,gy,gz\n")
        imu_csv_file.flush()

    audio_path = os.path.join(SESSION_DIR, "audio.csv")
    write_aud_hdr = not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0
    audio_csv_file = open(audio_path, "a", encoding="utf-8")
    if write_aud_hdr:
        audio_csv_file.write("timestamp_us,val\n")
        audio_csv_file.flush()

    temp_path = os.path.join(SESSION_DIR, "temperature.csv")
    write_tmp_hdr = not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0
    temp_csv_file = open(temp_path, "a", encoding="utf-8")
    if write_tmp_hdr:
        temp_csv_file.write("timestamp_ms,temp_object_c,temp_ambient_c\n")
        temp_csv_file.flush()

    print(f"[COLLECTION MODE ACTIVE] Recording raw sensor data to: {SESSION_DIR}", flush=True)

# ----------------- TWILIO ALERT HANDLER -----------------
last_twilio_alert_time = 0

def send_twilio_alert(similarity, anomaly_score, status, root_cause_str):
    global last_twilio_alert_time
    now = time.time()
    if now - last_twilio_alert_time < TWILIO_CFG.get("cooldown_seconds", 60):
        return
        
    last_twilio_alert_time = now
    target = TWILIO_CFG.get("target_phone", "+918401782327")
    sid = TWILIO_CFG.get("account_sid", "")
    token = TWILIO_CFG.get("auth_token", "")
    from_num = TWILIO_CFG.get("from_phone", "")
    
    msg_body = (
        f"[RETROFIT ALERT] Machine [{MACHINE_ID}] detected {status}!\n"
        f"Similarity: {similarity}%\n"
        f"Score: {anomaly_score}\n"
        f"Cause: {root_cause_str}\n"
        f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}"
    )
    
    if sid and not sid.startswith("AC_TWILIO") and token and not token.startswith("TWILIO_"):
        def _send():
            try:
                from urllib.parse import urlencode
                import urllib.request
                import base64
                
                url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
                data = urlencode({"To": target, "From": from_num, "Body": msg_body}).encode("utf-8")
                req = urllib.request.Request(url, data=data, method="POST")
                auth = base64.b64encode(f"{sid}:{token}".encode("utf-8")).decode("ascii")
                req.add_header("Authorization", f"Basic {auth}")
                with urllib.request.urlopen(req, timeout=8) as resp:
                    print(f"[TWILIO SUCCESS] SMS status {resp.status}", flush=True)
            except Exception as e:
                print(f"[TWILIO ERROR] {e}", flush=True)
        threading.Thread(target=_send, daemon=True).start()

# ----------------- NON-BLOCKING MQTT CLIENT -----------------
mqtt_client = None
mqtt_connected = False
mqtt_broker = config.get("mqtt_broker", "broker.emqx.io")
mqtt_port = int(config.get("mqtt_port", 1883))
mqtt_topic = config.get("mqtt_topic", f"retrofit/telemetry/{MACHINE_ID}")

if config.get("mqtt_enabled", True):
    try:
        import paho.mqtt.client as mqtt

        def _on_connect(client, userdata, flags, rc, *args):
            global mqtt_connected
            mqtt_connected = True
            print(f"[MQTT] CONNECTED to {mqtt_broker}:{mqtt_port}", flush=True)

        def _on_disconnect(client, userdata, *args):
            global mqtt_connected
            mqtt_connected = False

        try:
            mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except Exception:
            mqtt_client = mqtt.Client()

        mqtt_client.on_connect = _on_connect
        mqtt_client.on_disconnect = _on_disconnect
        mqtt_client.connect_async(mqtt_broker, mqtt_port, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[MQTT] Setup error: {e}", flush=True)

# ----------------- FAST SENSOR STATE -----------------
lock = threading.Lock()
imu_ring_buffer = []

latest_temp_obj = 30.0
latest_temp_amb = 29.5
latest_audio_peak = 1000

latest_ax = 0.02
latest_ay = 0.01
latest_az = -0.83
latest_gx = 0.0
latest_gy = 0.0
latest_gz = 0.0

# Calibrated Baselines & Deadbands for MAX9814 & MPU6050
VIB_NOISE_DEADBAND_G = 0.06      # Motion threshold (normal idle std is ~0.01-0.03g)
GYRO_NOISE_DEADBAND_DPS = 18.0   # Rotational threshold (idle gyro is ~1-5 dps)
SOUND_NOISE_DEADBAND_VOLTS = 1.3 # MAX9814 AGC noise floor is ~0.8V-1.1V pk-pk
TEMP_NOISE_DEADBAND_C = 4.0      # Thermal gradient threshold

smoothed_anomaly_score = 0.0

# ----------------- BACKGROUND WORKER THREAD -----------------
def background_inference_loop():
    global smoothed_anomaly_score, latest_ax, latest_ay, latest_az, latest_gx, latest_gy, latest_gz
    
    while True:
        try:
            time.sleep(0.2) # 5 Hz continuous inference cycle
            
            with lock:
                if len(imu_ring_buffer) >= 5:
                    win = list(imu_ring_buffer[-100:])
                    curr_ax, curr_ay, curr_az = latest_ax, latest_ay, latest_az
                    curr_gx, curr_gy, curr_gz = latest_gx, latest_gy, latest_gz
                    curr_temp_obj, curr_temp_amb = latest_temp_obj, latest_temp_amb
                    curr_sound_peak = latest_audio_peak
                    
                    ax_arr = np.array([r['ax'] for r in win], dtype=float) / 16384.0
                    ay_arr = np.array([r['ay'] for r in win], dtype=float) / 16384.0
                    az_arr = np.array([r['az'] for r in win], dtype=float) / 16384.0
                    gx_arr = np.array([r['gx'] for r in win], dtype=float) / 131.0
                    gy_arr = np.array([r['gy'] for r in win], dtype=float) / 131.0
                    gz_arr = np.array([r['gz'] for r in win], dtype=float) / 131.0
                    
                    acc_mag_win = np.sqrt(ax_arr**2 + ay_arr**2 + az_arr**2)
                    gyro_mag_win = np.sqrt(gx_arr**2 + gy_arr**2 + gz_arr**2)
                    
                    curr_vib_std = float(np.std(acc_mag_win))
                    curr_gyro_mean = float(np.mean(gyro_mag_win))
                else:
                    curr_ax, curr_ay, curr_az = latest_ax, latest_ay, latest_az
                    curr_gx, curr_gy, curr_gz = latest_gx, latest_gy, latest_gz
                    curr_temp_obj, curr_temp_amb = latest_temp_obj, latest_temp_amb
                    curr_vib_std = 0.01
                    curr_gyro_mean = 1.0
                    curr_sound_peak = 1000

            instant_acc_g = round(float(np.sqrt(curr_ax**2 + curr_ay**2 + curr_az**2)), 2)
            if instant_acc_g == 0.0:
                instant_acc_g = 0.83
                
            curr_sound_volts = round((curr_sound_peak * 3.3) / 16383.0, 3)
            curr_temp_delta = round(curr_temp_obj - curr_temp_amb, 2)
            
            # 1. Behavioral Fingerprint Statistical Distance to Trained Idle Laptop Profile
            # Target Idle Fingerprint: vib_std ~0.025g, temp_delta ~+1.8C, sound_volts ~0.45V, gyro ~2.5 dps
            TARGET_VIB_STD = 0.025
            TARGET_TEMP_DELTA = 1.80
            TARGET_SOUND_VOLTS = 0.45
            TARGET_GYRO_DPS = 2.5
            
            # Vibration distance: penalize missing fan vibration (table) and excess vibration (shake)
            if curr_vib_std < TARGET_VIB_STD:
                vib_dev = ((TARGET_VIB_STD - curr_vib_std) / 0.035) * 0.4 # Table penalty (~0.20)
            else:
                vib_dev = (curr_vib_std - TARGET_VIB_STD) / 0.080 # Excessive motion/shake penalty
                
            # Gyroscope distance (rotational movement)
            gyro_excess = max(0.0, curr_gyro_mean - 10.0)
            gyro_dev = gyro_excess / 35.0
            
            # Acoustic distance: relative departure from idle hum
            sound_diff = abs(curr_sound_volts - TARGET_SOUND_VOLTS)
            sound_dev = max(0.0, sound_diff - 0.40) / 0.80
            
            # Thermal distance: penalize missing laptop heat gradient (table) or overheating
            if curr_temp_delta < 0.3:
                temp_dev = ((TARGET_TEMP_DELTA - curr_temp_delta) / 2.0) * 0.45 # Table/Cool penalty (~0.40)
            else:
                temp_dev = max(0.0, (curr_temp_delta - TARGET_TEMP_DELTA) - 2.5) / 5.0
                
            # Raw composite behavioral distance
            raw_anomaly_score = float(0.40 * (vib_dev * 0.7 + gyro_dev * 0.3) + 0.25 * sound_dev + 0.35 * temp_dev)
            
            # 2. Smooth Persistence Filter (EMA)
            smoothed_anomaly_score = 0.65 * smoothed_anomaly_score + 0.35 * raw_anomaly_score
            score = round(smoothed_anomaly_score, 3)
            thresh = 1.0
            
            # Similarity curve: Idle laptop -> ~96-99%, Table -> ~68-76%, Shaking -> <35%
            similarity = float(np.clip(100.0 * (1.0 - (score / 1.60)), 0.0, 100.0))
            similarity = round(similarity, 1)
            
            if score > thresh * 1.1:
                status = "CRITICAL_ANOMALY"
            elif score > thresh * 0.55:
                status = "WARNING"
            else:
                status = "HEALTHY"
                
            total_dev = vib_dev + gyro_dev + sound_dev + temp_dev
            if total_dev > 0.001:
                vib_pct = round(float((vib_dev + gyro_dev) / total_dev * 100.0), 1)
                ac_pct  = round(float(sound_dev / total_dev * 100.0), 1)
                th_pct  = round(float(temp_dev / total_dev * 100.0), 1)
            else:
                vib_pct, ac_pct, th_pct = 33.3, 33.3, 33.4
                
            if curr_vib_std > 0.10 or gyro_dev > 0.4:
                top_cause = f"Excess Vibration Motion ({vib_pct}%)"
            elif temp_dev > 0.25 and curr_temp_delta < 0.3:
                top_cause = f"Thermal Gradient (Chassis Cool / Displaced {th_pct}%)"
            elif vib_dev > 0.15 and curr_vib_std < 0.012:
                top_cause = f"Vibration Floor (Fan Inactive / Displaced {vib_pct}%)"
            elif ac_pct >= vib_pct and ac_pct >= th_pct:
                top_cause = f"Acoustic Deviation ({ac_pct}%)"
            else:
                top_cause = "Operational Baseline Match" if status == "HEALTHY" else f"Behavioral Drift ({th_pct}%)"
                
            telemetry = {
                "timestamp_ms": int(time.time() * 1000),
                "machine_id": MACHINE_ID,
                "mode": "MONITORING",
                "operating_state": 0,
                "similarity_score": similarity,
                "anomaly_score": score,
                "status": status,
                "vibration_rms": instant_acc_g,
                "temperature_c": round(curr_temp_obj, 1),
                "sensors": {
                    "ax": curr_ax, "ay": curr_ay, "az": curr_az,
                    "gx": curr_gx, "gy": curr_gy, "gz": curr_gz,
                    "acc_vector_g": instant_acc_g,
                    "vib_std_g": round(curr_vib_std, 3),
                    "sound_volts": curr_sound_volts,
                    "temp_object_c": round(curr_temp_obj, 1),
                    "temp_ambient_c": round(curr_temp_amb, 1),
                    "temp_delta_c": curr_temp_delta
                },
                "modality_breakdown": {
                    "vibration_pct": vib_pct,
                    "acoustic_pct": ac_pct,
                    "thermal_pct": th_pct
                },
                "top_cause": top_cause
            }
            
            payload_str = json.dumps(telemetry)
            
            try:
                with open(os.path.join(DATA_DIR, "latest_health.json"), "w") as f:
                    f.write(payload_str)
            except Exception:
                pass
                
            if mqtt_client and mqtt_connected:
                try:
                    mqtt_client.publish(mqtt_topic, payload_str)
                except Exception:
                    pass
                    
            if status == "CRITICAL_ANOMALY" and TWILIO_CFG.get("enabled", True):
                send_twilio_alert(similarity, score, status, top_cause)

        except Exception as e:
            print(f"[WORKER ERROR] {e}", flush=True)

# Start Background Inference Worker
worker = threading.Thread(target=background_inference_loop, daemon=True)
worker.start()

# ----------------- LIGHTWEIGHT 0.1ms RPC CALLBACKS -----------------
def audio_batch(chunk: str):
    global latest_audio_peak
    if audio_csv_file and not audio_csv_file.closed:
        try:
            audio_csv_file.write(chunk if chunk.endswith("\n") else chunk + "\n")
            audio_csv_file.flush()
        except Exception:
            pass
            
    max_val, min_val = 0, 16383
    has_val = False
    for line in chunk.strip().split("\n"):
        if not line: continue
        parts = line.split(",")
        if len(parts) == 2:
            try:
                v = int(parts[1])
                if v > max_val: max_val = v
                if v < min_val: min_val = v
                has_val = True
            except ValueError:
                pass
    if has_val and max_val >= min_val:
        latest_audio_peak = max_val - min_val
    return "OK"

def imu_batch(chunk: str):
    global latest_ax, latest_ay, latest_az, latest_gx, latest_gy, latest_gz
    if imu_csv_file and not imu_csv_file.closed:
        try:
            imu_csv_file.write(chunk if chunk.endswith("\n") else chunk + "\n")
            imu_csv_file.flush()
        except Exception:
            pass
            
    lines = chunk.strip().split("\n")
    if not lines:
        return "OK"
        
    for line in lines:
        if not line: continue
        parts = line.split(",")
        if len(parts) == 7:
            try:
                ts_us = int(parts[0])
                raw_ax, raw_ay, raw_az = int(parts[1]), int(parts[2]), int(parts[3])
                raw_gx, raw_gy, raw_gz = int(parts[4]), int(parts[5]), int(parts[6])
                
                c_ax = round(raw_ax / 16384.0, 3)
                c_ay = round(raw_ay / 16384.0, 3)
                c_az = round(raw_az / 16384.0, 3)
                c_gx = round(raw_gx / 131.0, 1)
                c_gy = round(raw_gy / 131.0, 1)
                c_gz = round(raw_gz / 131.0, 1)
                
                with lock:
                    latest_ax, latest_ay, latest_az = c_ax, c_ay, c_az
                    latest_gx, latest_gy, latest_gz = c_gx, c_gy, c_gz
                    
                    imu_ring_buffer.append({
                        'ts_ms': ts_us // 1000,
                        'ax': raw_ax, 'ay': raw_ay, 'az': raw_az,
                        'gx': raw_gx, 'gy': raw_gy, 'gz': raw_gz,
                        'sound_peak': latest_audio_peak,
                        'ir_obj': latest_temp_obj,
                        'ir_amb': latest_temp_amb
                    })
                    if len(imu_ring_buffer) > 200:
                        imu_ring_buffer.pop(0)
            except Exception:
                pass
    return "OK"

def temp_row(line: str):
    global latest_temp_obj, latest_temp_amb
    if temp_csv_file and not temp_csv_file.closed:
        try:
            temp_csv_file.write(line if line.endswith("\n") else line + "\n")
            temp_csv_file.flush()
        except Exception:
            pass
            
    parts = line.split(",")
    if len(parts) == 3:
        try:
            latest_temp_obj = float(parts[1])
            latest_temp_amb = float(parts[2])
        except ValueError:
            pass
    return "OK"

Bridge.provide("imu_batch", imu_batch)
Bridge.provide("audio_batch", audio_batch)
Bridge.provide("temp_row", temp_row)

def periodic_loop():
    time.sleep(1.0)

App.run(user_loop=periodic_loop)
