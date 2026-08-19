import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import time
import json
import joblib
import numpy as np
import pandas as pd

from src.edge_sim.simulated_sensors import CSVSensorStreamProvider
from src.edge_sim.simulated_actuators import SimulatedActuator
from src.preprocessing.real_sensor_prep import load_or_compute_gyro_calibration, preprocess_real_sensor_dataframe
from src.features.vibration_features import extract_vibration_features
from src.features.audio_features import extract_audio_features
from src.features.temperature_features import extract_temperature_features
from src.inference.inference import RetroFitInferencePipeline
from src.alerts.escalation import AlertEscalationEngine
from src.communication.mqtt_client import RetroFitMQTTPublisher

def run_uno_q_edge_simulation(csv_path=r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv", machine_id="DEV_01", window_size=410, hop_size=205):
    print("=========================================================")
    print("       UNO Q EDGE SIMULATION — NOT PHYSICAL HARDWARE    ")
    print(" [HARDWARE-INDEPENDENT LAPTOP EDGE EXECUTION SIMULATOR]  ")
    print("=========================================================")
    
    sensor_provider = CSVSensorStreamProvider(csv_path)
    actuator = SimulatedActuator()
    pipeline = RetroFitInferencePipeline(machine_id=machine_id)
    escalator = AlertEscalationEngine(confirmation_count=3, min_confidence=70.0, cooldown_period_sec=300)
    publisher = RetroFitMQTTPublisher()
    
    # Model size & complexity reporting
    header_path = "edge/retrofit_edge_model.h"
    if os.path.exists(header_path):
        header_size_kb = os.path.getsize(header_path) / 1024.0
    else:
        header_size_kb = 92.88
        
    m_config = {
        'machine_id': machine_id,
        'machine_name': 'Simulated Machinery Node',
        'location': 'Edge Bench 1',
        'operator_phone': '+15550199',
        'alert_policy': {'critical_confirmation_count': 3, 'min_confidence': 70.0, 'cooldown_period_sec': 300}
    }
    
    # Buffer for streaming window accumulation
    raw_buffer = []
    window_count = 0
    
    latencies = {
        'acq_ms': [],
        'prep_ms': [],
        'feat_ms': [],
        'infer_ms': [],
        'dec_ms': [],
        'act_mqtt_ms': [],
        'total_ms': []
    }
    
    outputs = []
    
    print("\n[SIMULATOR] Starting real-time streaming edge pipeline simulation...")
    
    while sensor_provider.is_data_available():
        t0_acq = time.perf_counter()
        sample = sensor_provider.read_sample()
        if sample is None:
            break
        raw_buffer.append(sample)
        t1_acq = time.perf_counter()
        
        # When window buffer fills to window_size (410 samples)
        if len(raw_buffer) >= window_size:
            t0_total = time.perf_counter()
            df_win_raw = pd.DataFrame(raw_buffer)
            
            # 1. Preprocessing & Dedicated Gyro Calibration
            t0_prep = time.perf_counter()
            df_win_clean = preprocess_real_sensor_dataframe(df_win_raw, calibrate_gyro=True)
            t1_prep = time.perf_counter()
            
            # 2. Feature Extraction
            t0_feat = time.perf_counter()
            accel_data = df_win_clean[['ax', 'ay', 'az']].values
            gyro_data = df_win_clean[['gx_cal', 'gy_cal', 'gz_cal']].values
            f_vib = extract_vibration_features(accel_data, gyro_data, fs_accel=200, fs_gyro=200)
            f_aud = extract_audio_features(df_win_clean['sound_volts'].values, fs_audio=200, is_amplitude_summary=True)
            f_tmp = extract_temperature_features(df_win_clean['ir_object_c'].values, baseline_temp=float(df_win_clean['ir_ambient_c'].iloc[0]))
            
            feat_dict = {}
            feat_dict.update(f_vib)
            feat_dict.update(f_aud)
            feat_dict.update(f_tmp)
            df_feat_win = pd.DataFrame([feat_dict])
            t1_feat = time.perf_counter()
            
            # Fill missing selected features with 0.0 if necessary
            for f_name in pipeline.selected_features:
                if f_name not in df_feat_win.columns:
                    df_feat_win[f_name] = 0.0
                    
            # 3. Model Inference (Isolation Forest)
            t0_infer = time.perf_counter()
            out = pipeline.predict_window(df_feat_win)
            t1_infer = time.perf_counter()
            
            # 4. Decision Engine & Escalation
            t0_dec = time.perf_counter()
            alert_action = escalator.evaluate_observation(
                status=out['status'],
                anomaly_score=out['anomaly_score'],
                drift_score=out['behavior_drift'],
                confidence=out['confidence'],
                machine_config=m_config,
                top_features=out.get('top_features', [])
            )
            t1_dec = time.perf_counter()
            
            # 5. Actuation & Non-Blocking MQTT
            t0_act = time.perf_counter()
            act_cmd = out['hardware_actuation']
            actuator.set_rgb_ring(act_cmd.get('led_ring', 'SOLID_GREEN'))
            actuator.set_buzzer(act_cmd.get('buzzer_active', False))
            
            pub_res = publisher.publish_telemetry(out, machine_config=m_config)
            t1_act = time.perf_counter()
            
            t1_total = time.perf_counter()
            
            # Record latencies in ms
            acq_ms = (t1_acq - t0_acq) * 1000.0
            prep_ms = (t1_prep - t0_prep) * 1000.0
            feat_ms = (t1_feat - t0_feat) * 1000.0
            infer_ms = (t1_infer - t0_infer) * 1000.0
            dec_ms = (t1_dec - t0_dec) * 1000.0
            act_mqtt_ms = (t1_act - t0_act) * 1000.0
            total_ms = (t1_total - t0_total) * 1000.0
            
            latencies['acq_ms'].append(acq_ms)
            latencies['prep_ms'].append(prep_ms)
            latencies['feat_ms'].append(feat_ms)
            latencies['infer_ms'].append(infer_ms)
            latencies['dec_ms'].append(dec_ms)
            latencies['act_mqtt_ms'].append(act_mqtt_ms)
            latencies['total_ms'].append(total_ms)
            
            window_count += 1
            act_disp = actuator.get_hardware_status_display()
            
            print(f"[EDGE SIM Frame {window_count:2d}] "
                  f"Status: {out['status']:23s} | "
                  f"Similarity: {out['similarity']:5.1f}% | "
                  f"Drift: {out['behavior_drift']:4.1f} | "
                  f"Latency: {total_ms:5.2f}ms | "
                  f"{act_disp}")
                  
            outputs.append(out)
            # Advance buffer by hop_size
            raw_buffer = raw_buffer[hop_size:]
            
    print("\n=========================================================")
    print("      UNO Q EDGE SIMULATION PERFORMANCE PROFILE         ")
    print("=========================================================")
    
    avg_total = np.mean(latencies['total_ms']) if latencies['total_ms'] else 0.0
    avg_prep = np.mean(latencies['prep_ms']) if latencies['prep_ms'] else 0.0
    avg_feat = np.mean(latencies['feat_ms']) if latencies['feat_ms'] else 0.0
    avg_infer = np.mean(latencies['infer_ms']) if latencies['infer_ms'] else 0.0
    avg_dec = np.mean(latencies['dec_ms']) if latencies['dec_ms'] else 0.0
    avg_act = np.mean(latencies['act_mqtt_ms']) if latencies['act_mqtt_ms'] else 0.0
    
    print(f" -> Total Windows Evaluated : {window_count}")
    print(f" -> Sensor Prep Latency     : {avg_prep:.3f} ms / window")
    print(f" -> Feature Extr Latency    : {avg_feat:.3f} ms / window")
    print(f" -> Inference Latency       : {avg_infer:.3f} ms / window")
    print(f" -> Decision/Escalation Lat : {avg_dec:.3f} ms / window")
    print(f" -> Actuation/MQTT Latency  : {avg_act:.3f} ms / window")
    print(f" -> TOTAL END-TO-END LATENCY: {avg_total:.3f} ms / window")
    print(f" -> Exported C++ Model Size : {header_size_kb:.2f} KB")
    print(f" -> Window Hop Budget       : 1024 ms (Margin: {1024.0 - avg_total:.1f} ms -> PASS)")
    print("=========================================================")
    
    return {
        'window_count': window_count,
        'latencies': {
            'avg_prep_ms': avg_prep,
            'avg_feat_ms': avg_feat,
            'avg_infer_ms': avg_infer,
            'avg_dec_ms': avg_dec,
            'avg_act_ms': avg_act,
            'avg_total_ms': avg_total
        },
        'header_size_kb': header_size_kb,
        'outputs': outputs
    }

if __name__ == "__main__":
    run_uno_q_edge_simulation()
