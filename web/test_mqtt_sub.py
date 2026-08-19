import paho.mqtt.client as mqtt
import json
import time

BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "retrofit/telemetry/#"

print(f"Connecting to free EMQX broker @ {BROKER}:{PORT}...")

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[CONNECTED] Return code: {rc}. Subscribing to {TOPIC}...")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    print(f"\n[RECEIVED] Topic: {msg.topic}")
    try:
        data = json.loads(msg.payload.decode())
        print(json.dumps(data, indent=2))
    except Exception:
        print(msg.payload.decode())

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except Exception:
    client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)
client.loop_start()

print("Listening for 5 seconds...")
try:
    time.sleep(5)
except KeyboardInterrupt:
    pass
client.loop_stop()
client.disconnect()
print("Done.")
