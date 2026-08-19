import os
import json

VALID_LIFECYCLE_STATES = [
    "NEW",
    "REGISTERED",
    "CALIBRATION",
    "LEARNING",
    "BASELINE_NOT_READY",
    "BASELINE_READY",
    "FINGERPRINT_CREATED",
    "MONITORING"
]

class MachineRegistrationManager:
    """
    Manages machine-specific registration, configuration metadata, and lifecycle states.
    Machine type is descriptive metadata only; AI learns individual machine behavior from scratch.
    """
    def __init__(self, base_dir="data/machines"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def register_machine(self, machine_id, machine_name, machine_type, application="General", location="Facility 1", operator_phone="", alert_policy=None):
        machine_dir = os.path.join(self.base_dir, machine_id)
        os.makedirs(machine_dir, exist_ok=True)
        
        config = {
            'machine_id': machine_id,
            'machine_name': machine_name,
            'machine_type': machine_type,
            'application': application,
            'location': location,
            'operator_phone': operator_phone,
            'alert_policy': alert_policy or {
                'critical_confirmation_count': 3,
                'min_confidence': 70.0,
                'cooldown_period_sec': 300
            },
            'lifecycle_state': 'REGISTERED',
            'created_at': pd.Timestamp.now().isoformat() if 'pd' in globals() else ""
        }
        
        config_path = os.path.join(machine_dir, "machine_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"[REGISTRATION] Registered Machine '{machine_id}' ({machine_name}). Lifecycle: REGISTERED.")
        return config

    def get_machine_config(self, machine_id):
        config_path = os.path.join(self.base_dir, machine_id, "machine_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return None

    def update_lifecycle_state(self, machine_id, new_state):
        if new_state not in VALID_LIFECYCLE_STATES:
            raise ValueError(f"Invalid lifecycle state '{new_state}'. Allowed: {VALID_LIFECYCLE_STATES}")
            
        config = self.get_machine_config(machine_id)
        if not config:
            raise ValueError(f"Machine '{machine_id}' not found.")
            
        config['lifecycle_state'] = new_state
        config_path = os.path.join(self.base_dir, machine_id, "machine_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"[LIFECYCLE] Machine '{machine_id}' state updated to: {new_state}")
        return config

    def list_registered_machines(self):
        machines = []
        if os.path.exists(self.base_dir):
            for m_id in os.listdir(self.base_dir):
                cfg = self.get_machine_config(m_id)
                if cfg:
                    machines.append(cfg)
        return machines
