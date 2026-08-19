import json
import os
import time

class BehavioralMemory:
    """
    Manages historical event storage in data/memory/events.json.
    """
    def __init__(self, filepath="data/memory/events.json"):
        self.filepath = filepath
        self.events = []
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.events = json.load(f)
            except Exception:
                self.events = []

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self.events, f, indent=2)

    def add_event(self, operating_state, anomaly_score, behavior_drift, modality_contrib, top_features, embedding=None):
        event_id = f"EVT_{len(self.events)+1:03d}"
        event = {
            'event_id': event_id,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'operating_state': int(operating_state),
            'anomaly_score': float(anomaly_score),
            'behavior_drift': float(behavior_drift),
            'modality_contribution': modality_contrib,
            'top_features': top_features,
            'embedding': embedding if embedding is not None else []
        }
        self.events.append(event)
        self.save()
        return event
