class DecisionEngine:
    """
    Evaluates temporal persistence, anomaly status, and physical response trigger.
    """
    def __init__(self, persistence_n=4, persistence_m=6, warning_thresh=15.0, critical_thresh=40.0):
        self.n = persistence_n
        self.m = persistence_m
        self.warning_thresh = warning_thresh
        self.critical_thresh = critical_thresh
        self.history = []

    def evaluate(self, drift_score):
        self.history.append(drift_score)
        if len(self.history) > self.m:
            self.history.pop(0)
            
        warn_count = sum(1 for d in self.history if d >= self.warning_thresh)
        crit_count = sum(1 for d in self.history if d >= self.critical_thresh)
        
        if crit_count >= self.n:
            status = "CRITICAL_ANOMALY"
            led = "FLASHING_RED"
            buzzer = True
            relay = "OPEN"
        elif warn_count >= self.n:
            status = "UNKNOWN_UNSEEN_BEHAVIOR"
            led = "BREATHING_YELLOW"
            buzzer = False
            relay = "CLOSED"
        else:
            status = "KNOWN_NORMAL_STATE"
            led = "SOLID_GREEN"
            buzzer = False
            relay = "CLOSED"
            
        return {
            'status': status,
            'led_ring': led,
            'buzzer_active': buzzer,
            'relay_state': relay
        }
