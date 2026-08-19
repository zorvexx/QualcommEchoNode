from src.edge_sim.interfaces import IActuator

class SimulatedActuator(IActuator):
    """
    Simulates physical RGB LED ring and piezoelectric buzzer outputs.
    Outputs clear ASCII hardware status visualization in terminal.
    """
    def __init__(self):
        self.rgb_state = "SOLID_GREEN"
        self.buzzer_active = False
        self.pulse_pattern = "OFF"

    def set_rgb_ring(self, mode_str):
        self.rgb_state = mode_str

    def set_buzzer(self, active_bool, pulse_pattern="OFF"):
        self.buzzer_active = active_bool
        self.pulse_pattern = pulse_pattern

    def get_hardware_status_display(self):
        if self.rgb_state == "SOLID_GREEN":
            led_visual = "[LED: SOLID GREEN]"
        elif self.rgb_state == "BREATHING_YELLOW":
            led_visual = "[LED: BREATHING YELLOW]"
        elif self.rgb_state == "FLASHING_RED":
            led_visual = "[LED: FLASHING RED]"
        else:
            led_visual = "[LED: OFF]"
            
        buzzer_visual = "[BUZZER: ON]" if self.buzzer_active else "[BUZZER: OFF]"
        return f"[ACTUATION] {led_visual:24s} | {buzzer_visual}"
