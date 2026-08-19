import os
import argparse
import pandas as pd
import json
import yaml
from src.features.extractor import extract_features_from_dataframe
from src.inference.inference import RetroFitInferencePipeline

def main():
    parser = argparse.ArgumentParser(description="RetroFit Machine Behavioral Intelligence Pipeline CLI")
    parser.add_argument("--model", default="data/models", help="Directory containing trained model artifacts")
    parser.add_argument("--data", default="data/raw/demo_machine.csv", help="Input raw CSV data file")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    args = parser.parse_args()

    print("\n========================================================")
    print("      RETROFIT MACHINE BEHAVIORAL AI — INFERENCE ENGINE  ")
    print("========================================================")

    if not os.path.exists(args.model) or not os.path.exists(os.path.join(args.model, "anomaly_model.pkl")):
        print(f"[ERROR] Trained model artifacts not found in '{args.model}'. Please run scripts/train_model.py first.")
        return

    print(f"[1/3] Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"[2/3] Initializing RetroFit Inference Engine from {args.model}...")
    pipeline = RetroFitInferencePipeline(models_dir=args.model)

    print(f"[3/3] Reading input data from {args.data}...")
    df_raw = pd.read_csv(args.data)
    df_features = extract_features_from_dataframe(df_raw, config)

    # Perform inference on last window
    print(f"\nExecuting real-time inference on sample window frame...")
    last_window = df_features.tail(1)
    result = pipeline.predict_window(last_window)

    print("\n--- INFERENCE RESULT SUMMARY ---")
    print(json.dumps(result, indent=2))
    print("\n[SUCCESS] RetroFit inference completed successfully.")

if __name__ == "__main__":
    main()
