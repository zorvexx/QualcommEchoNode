import os
import json
import time
import queue
import threading

class RetroFitMQTTPublisher:
    """
    Handles non-blocking, asynchronous MQTT communication for RetroFit telemetry streaming.
    Worker thread processes publish queue in background so network latency never delays inference loop.
    Architecture: Uno Q -> MQTT -> Backend Subscriber -> Dashboard / Twilio.
    Includes safe offline/local fallback when external broker is unavailable.
    """
    def __init__(self, broker_host="localhost", broker_port=1883, topic_prefix="retrofit/telemetry"):
        self.broker_host = os.environ.get("MQTT_BROKER", broker_host)
        self.broker_port = int(os.environ.get("MQTT_PORT", broker_port))
        self.topic_prefix = topic_prefix
        self.connected = False
        self.client = None
        
        self.publish_queue = queue.Queue(maxsize=100)
        
        try:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client()
            self.client.connect_async(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            self.connected = True
            print(f"[MQTT] Initialized MQTT client targeting {self.broker_host}:{self.broker_port}.")
        except Exception as e:
            print(f"[MQTT WARNING] Could not connect to MQTT broker ({e}). Operating in Local Fallback Mode.")
            self.connected = False
            self.client = None
            
        # Start asynchronous background worker thread
        self.worker_thread = threading.Thread(target=self._async_worker, daemon=True)
        self.worker_thread.start()

    def _async_worker(self):
        """
        Background worker thread draining publish queue non-blockingly.
        """
        while True:
            try:
                topic, payload_str, payload = self.publish_queue.get(timeout=1.0)
                
                # Save latest telemetry to file for local UI streaming
                os.makedirs("data/models", exist_ok=True)
                with open("data/models/latest_telemetry.json", "w") as f:
                    json.dump(payload, f, indent=2)
                    
                if self.connected and self.client:
                    try:
                        self.client.publish(topic, payload_str)
                    except Exception:
                        pass
                self.publish_queue.task_done()
            except queue.Empty:
                pass
            except Exception:
                pass

    def publish_telemetry(self, inference_output, machine_config=None):
        """
        Non-blocking enqueue of telemetry payload.
        """
        machine_id = inference_output.get('machine_id', 'DEV_01')
        topic = f"{self.topic_prefix}/{machine_id}"
        
        payload = {
            'timestamp': inference_output.get('timestamp', time.time()),
            'machine_id': machine_id,
            'machine_name': machine_config.get('machine_name', 'Laptop') if machine_config else machine_id,
            'behavioral_cluster': inference_output.get('state', 0),
            'similarity': inference_output.get('similarity', 100.0),
            'anomaly_score': inference_output.get('anomaly_score', 0.0),
            'behavior_drift': inference_output.get('behavior_drift', 0.0),
            'confidence': inference_output.get('confidence', 99.0),
            'status': inference_output.get('status', 'KNOWN_NORMAL_STATE'),
            'modality_contributions': inference_output.get('modality_contribution', {}),
            'top_features': inference_output.get('top_features', []),
            'hardware_actuation': inference_output.get('hardware_actuation', {})
        }
        
        payload_str = json.dumps(payload)
        
        try:
            self.publish_queue.put_nowait((topic, payload_str, payload))
            return {'status': 'ENQUEUED_NON_BLOCKING', 'topic': topic, 'payload': payload}
        except queue.Full:
            return {'status': 'QUEUE_FULL_DROPPED', 'topic': topic, 'payload': payload}
