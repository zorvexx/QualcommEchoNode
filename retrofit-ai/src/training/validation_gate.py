from src.hardware.capabilities import HardwareCapabilityRegistry

def validate_hardware_feature_compatibility(feature_list, strict=True):
    """
    Pre-training validation gate.
    Verifies that all candidate or selected features are physically supported by the configured hardware node.
    If strict=True and invalid features are found, raises ValueError to abort training.
    """
    registry = HardwareCapabilityRegistry()
    is_valid, invalid_features = registry.validate_feature_set(feature_list)
    
    if not is_valid:
        msg = f"[HARDWARE VALIDATION GATE ERROR] Training aborted! Found {len(invalid_features)} physically unsupported features for the hardware configuration: {invalid_features}"
        print(msg)
        if strict:
            raise ValueError(msg)
        return False
        
    print(f"[HARDWARE VALIDATION GATE PASSED] All {len(feature_list)} features are physically valid for the sensor node.")
    return True
