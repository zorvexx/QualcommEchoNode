import os
import joblib
import json
import numpy as np
try:
    import torch
except ImportError:
    torch = None

def export_model_artifacts(model, scaler, selected_features, state_model, fingerprint_data, output_dir="data/models"):
    """
    Exports scaler, state model, anomaly model, and edge artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    edge_dir = os.path.join(output_dir, "..", "..", "edge")
    os.makedirs(os.path.join(edge_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(edge_dir, "config"), exist_ok=True)
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(scaler, os.path.join(edge_dir, "model", "scaler.pkl"))
    
    # Save State Model
    if state_model:
        joblib.dump(state_model, os.path.join(output_dir, "state_model.pkl"))
        joblib.dump(state_model, os.path.join(edge_dir, "model", "state_model.pkl"))
        
    # Save Selected Features
    with open(os.path.join(output_dir, "selected_features.json"), 'w') as f:
        json.dump(selected_features, f, indent=2)
    with open(os.path.join(edge_dir, "config", "selected_features.json"), 'w') as f:
        json.dump(selected_features, f, indent=2)
        
    # Save Anomaly Model
    if torch is not None and hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
        # PyTorch model -> export ONNX
        dummy_input = torch.randn(1, len(selected_features))
        onnx_path = os.path.join(output_dir, "anomaly_model.onnx")
        torch.onnx.export(model.model, dummy_input, onnx_path, input_names=['input'], output_names=['reconstructed', 'latent'])
        torch.onnx.export(model.model, dummy_input, os.path.join(edge_dir, "model", "anomaly_model.onnx"))
        joblib.dump(model, os.path.join(output_dir, "anomaly_model.pkl"))
    else:
        joblib.dump(model, os.path.join(output_dir, "anomaly_model.pkl"))
        joblib.dump(model, os.path.join(edge_dir, "model", "anomaly_model.pkl"))
        
    # Save Fingerprint
    with open(os.path.join(output_dir, "machine_fingerprint.json"), 'w') as f:
        json.dump(fingerprint_data, f, indent=2)
    with open(os.path.join(edge_dir, "config", "machine_fingerprint.json"), 'w') as f:
        json.dump(fingerprint_data, f, indent=2)
        
    print(f"[EXPORT] Artifacts exported to {output_dir} and {edge_dir}")
