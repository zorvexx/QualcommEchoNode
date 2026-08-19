from abc import ABC, abstractmethod

class ISensorProvider(ABC):
    """
    Abstract interface for sensor acquisition.
    """
    @abstractmethod
    def read_sample(self):
        """
        Reads a single raw sensor sample.
        Returns dict with keys: timestamp_ms, ax, ay, az, gx, gy, gz, sound_peak, sound_volts, ir_ambient_c, ir_object_c
        """
        pass
        
    @abstractmethod
    def is_data_available(self):
        """Returns True if samples remain in stream."""
        pass

class IActuator(ABC):
    """
    Abstract interface for hardware actuation (RGB LED ring, buzzer, relay).
    """
    @abstractmethod
    def set_rgb_ring(self, mode_str):
        """
        Sets RGB LED state: SOLID_GREEN, BREATHING_YELLOW, FLASHING_RED.
        """
        pass
        
    @abstractmethod
    def set_buzzer(self, active_bool, pulse_pattern="OFF"):
        """
        Sets Piezoelectric Buzzer state.
        """
        pass

class IEdgeModel(ABC):
    """
    Abstract interface for edge anomaly model evaluation.
    """
    @abstractmethod
    def predict_score(self, X_scaled):
        """
        Returns anomaly score for scaled feature vector.
        """
        pass
