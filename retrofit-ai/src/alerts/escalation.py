import os
import json
import time
from src.alerts.twilio_voice import TwilioVoiceNotifier

class AlertEscalationEngine:
    """
    Manages persistent critical-event confirmation, debounce, confidence filtering, and alert cooldowns.
    Prevents false alarms and repeated calls for ongoing anomalies.
    """
    def __init__(self, confirmation_count=3, min_confidence=70.0, cooldown_period_sec=300):
        self.confirmation_count = confirmation_count
        self.min_confidence = min_confidence
        self.cooldown_period_sec = cooldown_period_sec
        
        self.consecutive_critical_count = 0
        self.last_call_time = 0.0
        self.notifier = TwilioVoiceNotifier()

    def evaluate_observation(self, status, anomaly_score, drift_score, confidence, machine_config, top_features=None):
        """
        Evaluates real-time observation and executes escalation policy.
        """
        top_features = top_features or ['vibration_energy', 'temperature_rise']
        current_time = time.time()
        
        policy = machine_config.get('alert_policy', {})
        conf_req = policy.get('critical_confirmation_count', self.confirmation_count)
        min_conf_req = policy.get('min_confidence', self.min_confidence)
        cooldown_req = policy.get('cooldown_period_sec', self.cooldown_period_sec)
        phone = machine_config.get('operator_phone', '')
        
        action = {
            'status': status,
            'alert_level': 'INFO',
            'confirmed_critical': False,
            'twilio_call_triggered': False,
            'dashboard_warning': False,
            'message': 'Normal operation.'
        }
        
        # Reset consecutive count if status drops back to Normal or confidence is low
        if status == 'KNOWN_NORMAL_STATE' or confidence < min_conf_req:
            self.consecutive_critical_count = 0
            return action
            
        if status == 'UNKNOWN_UNSEEN_BEHAVIOR':
            self.consecutive_critical_count = 0
            action['alert_level'] = 'WARNING'
            action['dashboard_warning'] = True
            action['message'] = 'Unseen behavioral shift detected. Dashboard warning logged.'
            return action
            
        if status == 'CRITICAL_ANOMALY':
            self.consecutive_critical_count += 1
            action['alert_level'] = 'CRITICAL'
            action['dashboard_warning'] = True
            
            # Check Persistent Confirmation Count
            if self.consecutive_critical_count >= conf_req:
                action['confirmed_critical'] = True
                
                m_id = machine_config.get('machine_id', 'DEV_01')
                history_file = f"data/machines/{m_id}/alert_history.json"
                
                # Restore last call time from disk if available
                if os.path.exists(history_file):
                    try:
                        with open(history_file, 'r') as f:
                            hist_data = json.load(f)
                            self.last_call_time = hist_data.get('last_call_time', self.last_call_time)
                    except Exception:
                        pass
                        
                # Check Alert Cooldown
                time_since_last_call = current_time - self.last_call_time
                if time_since_last_call >= cooldown_req:
                    # Trigger Twilio Call
                    res = self.notifier.trigger_voice_alert(
                        to_phone=phone,
                        machine_name=machine_config.get('machine_name', 'Laptop'),
                        machine_id=m_id,
                        location=machine_config.get('location', 'Lab'),
                        drift_score=drift_score,
                        top_features=top_features
                    )
                    self.last_call_time = current_time
                    
                    # Persist alert history to disk
                    os.makedirs(os.path.dirname(history_file), exist_ok=True)
                    hist_log = []
                    if os.path.exists(history_file):
                        try:
                            with open(history_file, 'r') as f:
                                hist_log = json.load(f).get('events', [])
                        except Exception:
                            hist_log = []
                            
                    hist_log.append({'timestamp': current_time, 'drift': drift_score, 'res': res})
                    with open(history_file, 'w') as f:
                        json.dump({'last_call_time': self.last_call_time, 'events': hist_log}, f, indent=2)
                        
                    action['twilio_call_triggered'] = True
                    action['twilio_details'] = res
                    action['message'] = f"CONFIRMED CRITICAL ANOMALY! Escalated via Twilio ({res['mode']})."
                else:
                    action['message'] = f"CONFIRMED CRITICAL ANOMALY! Twilio call suppressed by cooldown ({int(cooldown_req - time_since_last_call)}s remaining)."
            else:
                action['message'] = f"Critical anomaly window detected ({self.consecutive_critical_count}/{conf_req} confirmations)."
                
        return action
