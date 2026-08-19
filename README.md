# RetroFit — Edge AI Machine Behavioral Intelligence System
> Non-Invasive Behavioral Fingerprinting for Machine Condition Monitoring & Predictive Maintenance  
> Developed for Arduino Uno Q (Qualcomm Linux Core + STM32 Zephyr RTOS)

---

## 1. Overview: "Is this machine behaving like itself?"
Traditional predictive maintenance systems attempt to compare every machine against generic, one-size-fits-all threshold tables. In real industrial environments, however, every motor, pump, and spindle has its own unique mechanical personality and operating baseline.

**RetroFit** takes a machine-first approach:
When attached to equipment with zero intrusive wiring, RetroFit observes its normal baseline multi-modal dynamics and learns an **8-dimensional behavioral fingerprint**. When bearings begin degrading, thermal heat builds up, or structural looseness occurs, the edge node identifies the statistical deviation and immediately alerts operators.

---

## 2. Hardware Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            ARDUINO UNO Q                │
                    │                                         │
 ┌───────────────┐  │  ┌──────────────────┐   Bridge RPC (UART)│  ┌──────────────────┐
 │ MPU6050 (IMU) │─►│  │ STM32U5 Core     │◄─────────────────►│  │ Qualcomm SoC      │
 ├───────────────┤  │  │ (Zephyr RTOS)    │                   │  │ (Linux / Python) │
 │ MAX9814 (Mic) │─►│  │                  │                   │  │                  │
 ├───────────────┤  │  │ • Sensor sampling│                   │  │ • Feature engine │
 │ MLX90614 (IR) │─►│  │ • Fast I2C & ADC │                   │  │ • ML inference   │
 └───────────────┘  │  └──────────────────┘                   │  │ • MQTT / Twilio  │
                    │                                         │  └─────────┬────────┘
                    └─────────────────────────────────────────┘            │
                                                                           │ WiFi / MQTT
                                                                           ▼
                                                               ┌──────────────────────┐
                                                               │ Live Web Dashboard   │
                                                               │ & Twilio SMS Alerts  │
                                                               └──────────────────────┘
```

---

## 3. Sensors & Modalities

* **MPU6050 (6-DoF IMU)**: 3-axis acceleration ($X, Y, Z$) and 3-axis angular velocity ($^\circ/\text{s}$) capturing structural vibration harmonics and mechanical shocks.
* **MAX9814 (Acoustic)**: Preamplified electret microphone with Automatic Gain Control (AGC) capturing high-frequency friction, squeaks, and acoustic pulses.
* **MLX90614 (Dual Infrared)**: Contactless thermal infrared measuring surface temperature ($T_{\text{object}}$) and room ambient temperature ($T_{\text{ambient}}$) to track thermal gradients ($\Delta T$).

---

## 4. Quick Start & Execution

### Running the System on Uno Q
```powershell
# 1. Flash firmware and start the Python engine
python deploy_to_unoq.py --mode MONITORING

# 2. Launch the local real-time web dashboard
# (or simply double-click web/index.html in any browser)
python web/serve_dashboard.py
```

---

## 5. Live Telemetry & Automated Alerts

### Real-Time Dashboard Stream
* **Broker**: `broker.emqx.io:1883` (MQTT TCP) / `port 8084` (Secure WebSockets)
* **Telemetry Topic**: `retrofit/telemetry/laptop_01`
* **Telemetry Rate**: 5 Hz continuous multi-rate stream (0ms visual latency)

### Twilio Alert Integration
When the system detects a critical behavioral anomaly, it dispatches an automated SMS alert:
* **Target Number**: `+91 84017 82327`
* **Cooldown Rate Limiting**: 60 seconds
* **Sample Alert Payload**:
```text
[RETROFIT ALERT] Machine [laptop_01] detected CRITICAL_ANOMALY!
Similarity: 28.4%
Score: 1.48
Cause: Excess Vibration Motion (78.2%)
Time: 22:15:30
```

---

## 6. Team & Collaborators
* **Om** — Lead Hardware & Edge Software Systems
* **Rushwa50** — ML Latent Fingerprinting & Autoencoder Architecture
* **Walterwhite10151** — Multi-Modal Model Evaluation & Validation
* **KahanMody15** — System Integration & Testing

