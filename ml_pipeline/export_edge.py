"""
RetroFit Edge Model Exporter for Arduino Uno Q
Converts Keras Autoencoder & Encoder to:
1. TensorFlow Lite (.tflite) for Linux/Python edge runtime on Uno Q Qualcomm core
2. C++ Byte Arrays (.h) for embedded MCU execution
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

def export_tflite_models(model_dir, output_dir=None):
    if output_dir is None:
        output_dir = model_dir
    os.makedirs(output_dir, exist_ok=True)
    
    ae_path = os.path.join(model_dir, "echonode_autoencoder.keras")
    enc_path = os.path.join(model_dir, "echonode_encoder.keras")
    
    # 1. Convert Autoencoder to TFLite
    ae_model = keras.models.load_model(ae_path)
    converter_ae = tf.lite.TFLiteConverter.from_keras_model(ae_model)
    converter_ae.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_ae = converter_ae.convert()
    
    ae_tflite_path = os.path.join(output_dir, "retrofit_autoencoder.tflite")
    with open(ae_tflite_path, "wb") as f:
        f.write(tflite_ae)
        
    # 2. Convert Encoder to TFLite
    enc_model = keras.models.load_model(enc_path)
    converter_enc = tf.lite.TFLiteConverter.from_keras_model(enc_model)
    converter_enc.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_enc = converter_enc.convert()
    
    enc_tflite_path = os.path.join(output_dir, "retrofit_encoder.tflite")
    with open(enc_tflite_path, "wb") as f:
        f.write(tflite_enc)
        
    print("\n[SUCCESS] Exported TensorFlow Lite models:")
    print(f"  -> {ae_tflite_path} ({len(tflite_ae)/1024:.1f} KB)")
    print(f"  -> {enc_tflite_path} ({len(tflite_enc)/1024:.1f} KB)")
    
    # 3. Export C++ Header for Arduino Embedded Runtime
    header_path = os.path.join(output_dir, "retrofit_model_data.h")
    with open(header_path, "w") as f:
        f.write("// RetroFit Auto-Generated Model Data for Arduino Uno Q\n")
        f.write("#ifndef RETROFIT_MODEL_DATA_H\n#define RETROFIT_MODEL_DATA_H\n\n")
        f.write(f"const unsigned int retrofit_autoencoder_len = {len(tflite_ae)};\n")
        f.write("const unsigned char retrofit_autoencoder_data[] = {\n  ")
        for i, b in enumerate(tflite_ae):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 16 == 0:
                f.write("\n  ")
        f.write("\n};\n\n")
        f.write("#endif // RETROFIT_MODEL_DATA_H\n")
        
    print(f"  -> C++ Header: {header_path}")
    return ae_tflite_path, enc_tflite_path
