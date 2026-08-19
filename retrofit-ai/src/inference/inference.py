import os
import joblib
import json
import numpy as np
import pandas as pd
from src.behavior.similarity import calculate_behavioral_similarity
from src.behavior.drift import BehaviorDriftTracker
from src.behavior.memory import BehavioralMemory
from src.behavior.historical_match import match_historical_events
from src.explainability.modality_contribution import compute_modality_contributions
from src.explainability.feature_contribution import compute_feature_contributions
from src.explainability.explanation import generate_machine_personality
from src.inference.decision_engine import DecisionEngine

class RetroFitInferencePipeline:
    """
    Complete real-time or offline batch inference pipeline.
    """
    def __init__(self, machine_id=None, models_dir="data/models", base_machines_dir="data/machines"):
        self.machine_id = machine_id
        
        # Route to machine-specific directory if machine_id provided and exists
        if machine_id and os.path.exists(os.path.join(base_machines_dir, machine_id, "anomaly_model.pkl")):
            self.models_dir = os.path.join(base_machines_dir, machine_id)
        elif machine_id and os.path.exists(os.path.join(models_dir, machine_id, "anomaly_model.pkl")):
            self.models_dir = os.path.join(models_dir, machine_id)
        else:
            self.models_dir = models_dir
            
        print(f"[INFERENCE PIPELINE] Loading machine artifacts from {self.models_dir}")
        self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
        self.anomaly_model = joblib.load(os.path.join(self.models_dir, "anomaly_model.pkl"))
        
        state_path = os.path.join(self.models_dir, "state_model.pkl")
        self.state_model = joblib.load(state_path) if os.path.exists(state_path) else None
        
        with open(os.path.join(self.models_dir, "selected_features.json"), 'r') as f:
            self.selected_features = json.load(f)
            
        with open(os.path.join(self.models_dir, "machine_fingerprint.json"), 'r') as f:
            self.fingerprint = json.load(f)
            
        self.threshold = float(self.fingerprint.get('anomaly_threshold', 1.0))
        self.state_thresholds = self.fingerprint.get('state_thresholds', {})
        self.drift_tracker = BehaviorDriftTracker(alpha=0.2)
        self.memory = BehavioralMemory()
        self.decision_engine = DecisionEngine()

    def predict_window(self, df_window_features):
        """
        Executes complete inference pipeline on single feature vector dataframe.
        """
        X = df_window_features[self.selected_features]
        X_scaled = self.scaler.transform(X)
        
        # 1. State prediction
        state_id = int(self.state_model.predict(X_scaled)[0]) if self.state_model else 0
        
        # 2. Anomaly score
        anomaly_score = float(self.anomaly_model.predict_score(X_scaled)[0])
        
        # State-specific threshold resolution
        thresh = float(self.state_thresholds.get(str(state_id), self.threshold))
        
        # Check feature-level Z-score deviation relative to healthy baseline
        feat_means = np.array([self.fingerprint['feature_means'].get(f, 0.0) for f in self.selected_features])
        feat_stds = np.array([self.fingerprint['feature_stds'].get(f, 1.0) for f in self.selected_features])
        feat_z_scores = np.abs(X.values[0] - feat_means) / (feat_stds + 1e-6)
        max_z_score = float(np.max(feat_z_scores))
        
        # If individual feature exceeds 4.0 Z-scores, scale effective score
        if max_z_score > 4.0 and anomaly_score < thresh:
            effective_score = thresh * (max_z_score / 4.0)
        else:
            effective_score = anomaly_score
        
        # Three-Tier Taxonomy Status Override
        if effective_score > 3.0 * thresh or max_z_score > 4.0:
            status_override = "CRITICAL_ANOMALY"
        elif effective_score > thresh:
            status_override = "UNKNOWN_UNSEEN_BEHAVIOR"
        else:
            status_override = None
            
        # 3. Similarity & Drift
        similarity = calculate_behavioral_similarity(effective_score, thresh)
        inst_drift, smoothed_drift = self.drift_tracker.update(similarity)
        
        # 4. Feature residuals & Modality attribution
        if hasattr(self.anomaly_model, 'get_feature_residuals'):
            residuals = self.anomaly_model.get_feature_residuals(X_scaled)[0]
        else:
            diff = (X_scaled[0] - self.fingerprint['feature_means'].get(self.selected_features[0], 0))**2
            residuals = np.ones(len(self.selected_features)) * anomaly_score
            
        modality_contrib = compute_modality_contributions(self.selected_features, residuals)
        top_feats, feat_contribs = compute_feature_contributions(self.selected_features, residuals, top_k=3)
        
        # 5. Historical memory matching
        latent = None
        if hasattr(self.anomaly_model, 'get_latent_embeddings'):
            latent = self.anomaly_model.get_latent_embeddings(X_scaled)[0]
            
        matches = match_historical_events(self.memory.events, latent)
        
        # Add event to memory if drift is high and not matched
        if smoothed_drift > 25.0 and not matches:
            self.memory.add_event(state_id, anomaly_score, smoothed_drift, modality_contrib, top_feats, latent.tolist() if latent is not None else None)
            
        # 6. Decision Engine
        decision = self.decision_engine.evaluate(smoothed_drift)
        if status_override:
            decision['status'] = status_override
            
        confidence = round(max(50.0, min(99.0, 100.0 - abs(smoothed_drift - inst_drift))), 1)
        personality = generate_machine_personality(self.fingerprint, feat_contribs)
        
        output = {
            'machine_id': self.fingerprint.get('machine_id', 'MACHINE_01'),
            'state': state_id,
            'similarity': round(similarity, 1),
            'behavior_drift': round(smoothed_drift, 1),
            'anomaly_score': round(anomaly_score, 4),
            'confidence': confidence,
            'status': decision['status'],
            'modality_contribution': modality_contrib,
            'top_features': top_feats,
            'feature_contributions': feat_contribs,
            'historical_matches': matches,
            'machine_personality': personality,
            'hardware_actuation': decision
        }
        return output
