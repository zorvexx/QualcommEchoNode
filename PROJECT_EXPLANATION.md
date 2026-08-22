# 🧠 RetroFit (Qualcomm RetroFit) — The Complete Plain-English Guide

> **Welcome to the Project Guide!**  
> This document explains **everything** about this project from the ground up. Whether you are presenting to judges, explaining it to your teammates, or studying how the code and hardware work together, this guide breaks down every concept without confusing math or ML jargon.

---

## 📑 Table of Contents
1. [The Big Picture: What Problem Does This Solve?](#1-the-big-picture-what-problem-does-this-solve)
2. [The Hardware: A "Two-Brain" Computer](#2-the-hardware-a-two-brain-computer)
3. [The Sensor Trio: How It Senses the World](#3-the-sensor-trio-how-it-senses-the-world)
4. [The Machine Learning: Explained Simply](#4-the-machine-learning-explained-simply)
5. [Root-Cause Attribution: How the AI Diagnoses Problems](#5-root-cause-attribution-how-the-ai-diagnoses-problems)
6. [Data Pipeline: How Data Moves in Real Time](#6-data-pipeline-how-data-moves-in-real-time)
7. [Twilio Emergency Alert System](#7-twilio-emergency-alert-system)
8. [Second-by-Second Lifecycle: What Happens When You Shake the Board?](#8-second-by-second-lifecycle)
9. [Key Questions & Answers for Judges / Presentations](#9-key-questions--answers-for-judges--presentations)

---

## 1. The Big Picture: What Problem Does This Solve?

### The Problem in Real Factories:
In modern factories, critical machines (motors, pumps, conveyor belts, CNC machines, transformers) break down unexpectedly. When a machine breaks down without warning:
- The whole production line stops (costing thousands of dollars per minute).
- Replacing entire factory machines with "smart modern machines" costs millions.

### The RetroFit Solution:
Instead of buying new machines, we **"Retrofit"** (stick) a small, low-cost intelligent AI node onto any old, dumb machine. 
- The node **learns what the machine feels, sounds, and heats like when running normally**.
- The moment the machine starts vibrating slightly differently, heating up abnormally, or making squealing friction sounds, the AI **detects the behavioral drift days or weeks before catastrophic breakdown**.
- It instantly alerts maintenance engineers on their mobile phones via **Twilio SMS** and visualizes the condition in real time on a live web dashboard.

---

## 2. The Hardware: A "Two-Brain" Computer

Our system runs on the **Arduino Uno Q (powered by Qualcomm & STMicroelectronics)**. It is unique because it contains **two separate processors (brains) on a single board**:

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                           ARDUINO UNO Q BOARD                          │
 │                                                                        │
 │  ┌───────────────────────────┐         ┌────────────────────────────┐  │
 │  │   STM32 Microcontroller   │         │    Qualcomm Linux Core     │  │
 │  │    (The "Spinal Cord")    │         │     (The "AI Brain")       │  │
 │  │                           │  UART   │                            │  │
 │  │ • Sub-millisecond reads   │ ◄─────► │ • Python 3.13 Engine       │  │
 │  │ • I2C & Analog hardware   │  Bridge │ • Neural Networks / ML     │  │
 │  │ • Never drops samples     │         │ • MQTT Cloud Streaming     │  │
 │  │ • Ultra-low power         │         │ • Twilio SMS Dispatch      │  │
 │  └─────────────┬─────────────┘         └──────────────┬─────────────┘  │
 └────────────────┼──────────────────────────────────────┼────────────────┘
                  │                                      │
          [Sensors Attached]                     [WiFi / Internet]
```

### Why Do We Need Two Brains?
1. **Brain 1: The STM32 Microcontroller (`sketch.ino`)**:
   - Think of this as the **spinal cord / nervous system**.
   - It is dedicated to one single job: reading sensor pins at lightning speed (50–500 times every second) without ever getting interrupted or delayed by operating system tasks.
2. **Brain 2: The Qualcomm Linux Core (`edge_main.py`)**:
   - Think of this as the **cerebral cortex / thinking brain**.
   - It runs full Linux and Python, manages complex Machine Learning math, calculates statistical deviations, connects to WiFi, and handles cloud communications.
3. **The Communication Bridge (`Arduino_RouterBridge`)**:
   - The two brains talk to each other over an internal high-speed serial link using Remote Procedure Calls (RPC). The STM32 bundles sensor readings into small batches and hands them over to Python.

---

## 3. The Sensor Trio: How It Senses the World

To truly understand a machine's behavior, we use **3 physical modalities** (Vibration, Sound, and Temperature):

```
       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
       │     MPU6050     │       │     MAX9814     │       │    MLX90614     │
       │ Vibration & Gyro│       │ Acoustic Noise  │       │ Thermal Infrared│
       └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
                │                         │                         │
      [3-Axis Accel + Gyro]       [Peak-to-Peak Sound]       [Object & Ambient Temp]
                │                         │                         │
                └────────────────►  DATA FUSION  ◄──────────────────┘
```

| Sensor | What It Measures | Physical Principle | Real-Life Analogy |
| :--- | :--- | :--- | :--- |
| **MPU6050** | 3-Axis Acceleration ($X, Y, Z$) & 3-Axis Gyroscope Rotation | Measures gravitational force ($g$) and angular speed ($^\circ/\text{s}$). | Feeling the rumble or shaking of an engine with your hand. |
| **MAX9814** | Acoustic Sound & Noise Volts | Electret microphone with Automatic Gain Control (AGC) measuring sound waveform peak-to-peak amplitude. | A doctor using a stethoscope to listen to heartbeat irregularities or friction squeaks. |
| **MLX90614** | Contactless Thermal Infrared ($T_{\text{object}}$, $T_{\text{ambient}}$, $\Delta T$) | Measures emitted infrared radiation from the machine surface without touching it, compared to room air. | Holding your palm near a hot radiator to sense heat build-up. |

---

## 4. The Machine Learning: Explained Simply

Many traditional monitoring systems use dumb static limits (e.g. *"Alarm if temperature > 50°C"*).  
**Why that fails:** In the heat of summer, 50°C might be normal; in the freezing winter, 50°C might mean a fire! If the machine is turned off, a static rule won't know it's disconnected.

### The Behavioral Fingerprint Concept:
Every machine has a unique **"Behavioral Fingerprint"** when running in its normal state:
- **Idle Laptop Example:**
  - **Vibration:** Has a continuous micro-tremor from the cooling fan motor ($\approx 0.020g - 0.035g$).
  - **Thermal:** The processor heats the chassis slightly above room temperature ($\Delta T = T_{\text{surface}} - T_{\text{ambient}} \approx +1.5^\circ\text{C} \text{ to } +3.0^\circ\text{C}$).
  - **Acoustic:** Has a continuous low-level fan hum ($\approx 0.35\text{V} - 0.50\text{V}$).

### How the AI Evaluates Health (The 3 States):

```
       [Machine Sensor Feed]
                 │
                 ▼
 ┌───────────────────────────────┐
 │   Statistical Feature Engine  │ ──► Extracts: RMS Vibration, Gyro Mean,
 └───────────────┬───────────────┘               Sound Amplitude, Thermal Delta (ΔT)
                 │
                 ▼
 ┌───────────────────────────────┐
 │ Behavioral Distance Matching  │ ──► Compares live signature to trained baseline
 └───────────────┬───────────────┘
                 │
        ┌────────┴───────────────────────────┐
        ▼                                    ▼                                    ▼
┌──────────────────┐               ┌──────────────────┐                 ┌──────────────────┐
│   ON LAPTOP IDLE │               │     ON TABLE     │                 │ SHAKEN / DAMAGE  │
│                  │               │                  │                 │                  │
│ • Fan Hum: Match │               │ • Fan Hum: NONE  │                 │ • Vibration: >>> │
│ • Heat ΔT: Match │               │ • Heat ΔT: NONE  │                 │ • Gyro: Spiking  │
│                  │               │                  │                 │                  │
│  Similarity: 98% │               │  Similarity: 87% │                 │  Similarity: 25% │
│  State: HEALTHY  │               │  State: HEALTHY  │                 │  State: CRITICAL │
└──────────────────┘               └──────────────────┘                 └────────┬─────────┘
                                                                                 │
                                                                                 ▼
                                                                        [Twilio SMS Dispatched]
```

1. **When on the Idle Laptop (`Similarity: 96% - 99%`)**:
   - The fan vibrations and temperature difference match the trained profile.
   - Status: **`HEALTHY`** (Green).

2. **When placed on a Flat Table (`Similarity: ~85% - 88%`)**:
   - The board is totally stationary, but the laptop fan micro-vibrations and heat source are absent.
   - The AI notices the deviation and identifies: **"Chassis Cool / Machine Displaced"**.

3. **When Shaken, Dropped, or Perturbed (`Similarity: < 35%`)**:
   - High $g$-force ($>1.5g$) and rotational velocity break through the safety gates.
   - Status: **`CRITICAL_ANOMALY`** (Red) $\rightarrow$ Anomaly Score rises above $1.0$.

---

## 5. Root-Cause Attribution: How the AI Diagnoses Problems

When an anomaly occurs, factory engineers do not just want to know *"something is wrong"*. They want to know **WHAT is wrong**.

Our system uses **Modal Feature Decomposition**:
$$\text{Total Deviation} = \text{Deviation}_{\text{Vibration}} + \text{Deviation}_{\text{Acoustic}} + \text{Deviation}_{\text{Thermal}}$$

$$\text{Attribution \%} = \frac{\text{Deviation}_{\text{Modality}}}{\text{Total Deviation}} \times 100$$

### Real-World Diagnostic Examples:
- **Bearing Wear / Loose Bolt:** Vibration deviation spikes to 80% $\rightarrow$ Dashboard displays: `Top Cause: Excess Vibration Motion (80%)`.
- **Fan Motor Failure / Blocked Air Duct:** Thermal delta spikes to 75% $\rightarrow$ Dashboard displays: `Top Cause: Thermal Gradient Overheat (75%)`.
- **Friction / Acoustic Screech:** Acoustic sensor detects high voltage amplitude $\rightarrow$ Dashboard displays: `Top Cause: Acoustic Spike (90%)`.

---

## 6. Data Pipeline: How Data Moves in Real Time

```
  [Physical Machine]
          │
          ▼
   (MPU + MIC + MLX)
          │  (I2C / ADC @ 50-500 Hz)
          ▼
  [STM32 Firmware] ──(Bridge RPC)──► [Python Engine (Edge AI)]
                                              │
                                              ▼  (MQTT TCP Port 1883)
                                     [EMQX Cloud Broker]
                                              │
                                              ▼  (Secure WebSockets Port 8084)
                                    [Live Web Dashboard (Chart.js)]
```

### Why MQTT & WebSockets?
1. **MQTT (`broker.emqx.io:1883`)**:
   - Ultra-lightweight binary protocol built specifically for IoT.
   - Uses a Publish/Subscribe pattern: the Uno Q publishes to `retrofit/telemetry/laptop_01`, and any authorized screen in the world can subscribe.
2. **WebSockets in the Browser (`wss://broker.emqx.io:8084/mqtt`)**:
   - Standard HTTP requires the browser to constantly reload/ask for new data.
   - WebSockets keep a permanent two-way pipe open. The moment the Uno Q emits a packet, the web page updates with **zero visual lag (0ms)**.

---

## 7. Twilio Emergency Alert System

When an anomaly crosses the critical threshold ($\text{Anomaly Score} > 1.1$), relying on someone staring at a screen is not enough.

### Automated SMS Dispatch:
```
                      ┌───────────────────────────────┐
                      │    CRITICAL ANOMALY EVENT     │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    Twilio Anti-Spam Check     │
                      │  (Has 60s elapsed since last?)│
                      └───────────────┬───────────────┘
                                      │ YES
                                      ▼
                      ┌───────────────────────────────┐
                      │ HTTPS POST to api.twilio.com  │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                         [Engineer's Mobile Phone]
                       +---------------------------+
                       | [RETROFIT ALERT]          |
                       | Machine: laptop_01        |
                       | Status: CRITICAL_ANOMALY  |
                       | Similarity: 24.5%         |
                       | Cause: Excess Vibration   |
                       | Time: 22:15:30            |
                       +---------------------------+
```

---

## 8. Second-by-Second Lifecycle

Here is the exact journey of a single vibration shock:

1. **$t = 0\text{ ms}$**: You tap the table or shake the laptop.
2. **$t = 20\text{ ms}$**: The MPU6050 accelerometer registers a physical spike of $1.85g$.
3. **$t = 40\text{ ms}$**: STM32 buffers 5 readings and sends an RPC chunk via `Bridge.call("imu_batch")` to the Qualcomm Linux processor.
4. **$t = 45\text{ ms}$**: Python background inference worker reads the buffer, calculates rolling RMS standard deviation, and identifies an excess disturbance.
5. **$t = 60\text{ ms}$**: The AI drops the Behavioral Similarity from $98\%$ to $25\%$ and marks Status = `CRITICAL_ANOMALY`.
6. **$t = 70\text{ ms}$**: Python publishes a JSON telemetry payload to MQTT broker `broker.emqx.io`.
7. **$t = 95\text{ ms}$**: Browser WebSocket receives the JSON; Chart.js smoothly animates the spike on screen.
8. **$t = 120\text{ ms}$**: Twilio REST API request is dispatched in a background thread to send an SMS to `+91 84017 82327`.

---

## 9. Key Questions & Answers for Presentations / Judges

### Q1: "Why do inference on the Edge (Uno Q) instead of sending all raw data to the Cloud?"
* **Answer:** 
  1. **Bandwidth:** Sending raw audio (8 kHz) and IMU (200 Hz) to the cloud 24/7 consumes hundreds of gigabytes per month per machine. On the edge, we only send a tiny summary packet (500 bytes) at 5 Hz.
  2. **Latency & Safety:** If a machine is about to tear itself apart, an edge node reacts in **milliseconds** without needing an active internet connection.
  3. **Privacy & Security:** Sensitive acoustic sounds from inside a factory never leave the board.

### Q2: "How does the system distinguish between normal machine operation and anomalies?"
* **Answer:** By fusing **three complementary physical modalities** (Vibration + Sound + Temperature). A single sensor might produce false positives (e.g., a door slamming nearby), but our multi-modal fusion checks if the physical vibration signature matches the machine's thermal and acoustic fingerprint.

### Q3: "Can this system be used on different types of industrial equipment?"
* **Answer:** Yes! The entire pipeline is agnostic to the machine type. By collecting a 2-minute baseline recording of any motor, pump, compressor, or vehicle engine, the system builds a custom behavioral fingerprint profile for that specific asset.

---

### 👥 Team Credits & Collaborators
- **Project:** RetroFit / Qualcomm RetroFit
- **Target Platform:** Arduino Uno Q (Qualcomm Linux + STM32 Zephyr RTOS)
- **Collaborators:** Rushwa50, Walterwhite10151, KahanMody15
