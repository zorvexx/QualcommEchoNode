#!/usr/bin/env python3
"""
RetroFit Serial Collector - Runs DIRECTLY on Uno Q host (NOT in container).
Reads /dev/ttyHS1 at 115200 baud and writes IMU/Audio/Temp CSVs.

Usage (run on Uno Q via SSH):
    sudo python3 /tmp/collect.py cpu_new pure_idle_04
"""
import sys, os, time, datetime, signal, threading

BAUD = 115200
SERIAL_PORT = "/dev/ttyHS1"
DATA_BASE = "/home/arduino/ArduinoApps/retrofit/python/data"

def main():
    machine_id = sys.argv[1] if len(sys.argv) > 1 else "machine_01"
    session_id = sys.argv[2] if len(sys.argv) > 2 else "session_01"

    session_dir = os.path.join(DATA_BASE, f"{machine_id}_{session_id}")
    os.makedirs(session_dir, exist_ok=True)

    imu_path  = os.path.join(session_dir, "imu.csv")
    aud_path  = os.path.join(session_dir, "audio.csv")
    tmp_path  = os.path.join(session_dir, "temperature.csv")

    print(f"[COLLECT] Session: {session_dir}", flush=True)
    print(f"[COLLECT] Opening {SERIAL_PORT} @ {BAUD} baud...", flush=True)

    try:
        import serial
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    except ImportError:
        # Fallback: use raw open if pyserial not available
        ser = open(SERIAL_PORT, "rb")
        ser.read_line = lambda: ser.readline()

    # Wait for READY signal
    print("[COLLECT] Waiting for RETROFIT_SERIAL_READY...", flush=True)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if "RETROFIT_SERIAL_READY" in line:
                print("[COLLECT] STM32 ready! Starting data capture.", flush=True)
                break
        except Exception:
            pass

    imu_rows = 0
    aud_rows = 0
    tmp_rows = 0
    start = time.time()

    with open(imu_path, "w") as fi, open(aud_path, "w") as fa, open(tmp_path, "w") as ft:
        fi.write("timestamp_us,ax,ay,az,gx,gy,gz\n"); fi.flush()
        fa.write("timestamp_us,val\n"); fa.flush()
        ft.write("timestamp_ms,temp_object_c,temp_ambient_c\n"); ft.flush()

        print(f"[COLLECT] Recording... Ctrl+C to stop.", flush=True)
        try:
            while True:
                try:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    parts = line.split(",")
                    kind = parts[0]

                    if kind == "I" and len(parts) == 8:
                        fi.write(f"{parts[1]},{parts[2]},{parts[3]},{parts[4]},{parts[5]},{parts[6]},{parts[7]}\n")
                        imu_rows += 1
                        if imu_rows % 500 == 0:
                            fi.flush()
                            elapsed = time.time() - start
                            print(f"[COLLECT] t={elapsed:.0f}s  IMU={imu_rows}  AUD={aud_rows}  TMP={tmp_rows}", flush=True)

                    elif kind == "A" and len(parts) == 3:
                        fa.write(f"{parts[1]},{parts[2]}\n")
                        aud_rows += 1
                        if aud_rows % 5000 == 0:
                            fa.flush()

                    elif kind == "T" and len(parts) == 4:
                        ft.write(f"{parts[1]},{parts[2]},{parts[3]}\n")
                        ft.flush()
                        tmp_rows += 1

                except Exception as e:
                    pass

        except KeyboardInterrupt:
            pass

    ser.close()
    print(f"\n[COLLECT] DONE. IMU={imu_rows} rows  AUD={aud_rows} rows  TMP={tmp_rows} rows", flush=True)
    print(f"[COLLECT] Files saved to: {session_dir}", flush=True)

if __name__ == "__main__":
    main()
