# RetroFit Machine Behavioral AI Engine
**Arduino Physical AI Challenge India 2026**

RetroFit is a self-learning machine behavioral intelligence system for the Arduino Uno Q.
Instead of relying on generic pre-trained static health classifiers, RetroFit:
1. Learns the normal behavioral identity of an individual physical machine under healthy conditions.
2. Discovers its normal operating states (Idle, Startup, Steady Load, Peak Load).
3. Continuously measures behavioral drift and similarity.
4. Explains which sensor modalities and features caused deviations.
5. Remembers historical machine behavior in an embedded vector memory database.
6. Generates interpretable Machine Intelligence Reports.

---

## 1. Quick Start Execution Commands

### Environment Setup
```bash
# Activate virtual environment
.venv/Scripts/activate
```

### Step 1: Generate Realistic Synthetic Demo Sensor Data
```bash
python scripts/generate_demo_data.py --output data/raw/demo_machine.csv --duration 600
```

### Step 2: Run End-to-End Feature Extraction, State Discovery & Model Training
```bash
python scripts/train_model.py --data data/raw/demo_machine.csv
```

### Step 3: Run Model Evaluation Benchmark
```bash
python scripts/evaluate_model.py --data data/raw/demo_machine.csv
```

### Step 4: Run Real-Time / Batch Inference CLI
```bash
python main.py --data data/raw/demo_machine.csv --model data/models/
```

### Step 5: Generate Machine Intelligence PDF Report
```bash
python scripts/generate_report.py
```

### Step 6: Run Automated Test Suite
```bash
python -m unittest tests/test_pipeline.py
```

---

## 2. Directory Structure
```
retrofit-ai/
├── data/ (raw, processed, features, models, memory, reports)
├── src/
│   ├── preprocessing/ (vibration, audio, temperature, synchronization)
│   ├── features/ (statistical, spectral, temporal, wavelet, extractors)
│   ├── selection/ (importance, permutation, selector)
│   ├── states/ (state_discovery)
│   ├── models/ (isolation_forest, mahalanobis, one_class_svm, autoencoder, model_comparison)
│   ├── behavior/ (fingerprint, similarity, drift, memory, historical_match)
│   ├── explainability/ (modality_contribution, feature_contribution, explanation)
│   ├── training/ (prepare_dataset, train, validate, export_model)
│   └── inference/ (inference, decision_engine)
├── edge/ (model, config, README.md)
├── scripts/ (generate_demo_data, extract_features, train_model, evaluate_model, generate_report)
├── tests/ (test_pipeline.py)
├── config.yaml
├── requirements.txt
├── README.md
└── main.py
```

---

## 3. Philosophy
**SENSE → LEARN → BUILD IDENTITY → UNDERSTAND STATES → MONITOR → DETECT DRIFT → EXPLAIN → REMEMBER → DECIDE → ACT**
