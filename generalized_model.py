import os
import numpy as np
import tensorflow as tf

def load_generalized_model():
    """
    Load the custom-trained ResNet50 model for semantic AI image detection.
    """
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "models", "semantic_vision_best.h5")
    
    # If the user hasn't trained it yet, return None so it doesn't crash
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Please run train_generalized.py first.")
        return None
        
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def run_generalized(img_pil, model):
    """
    Run inference using the custom ResNet50 generalized model.
    Returns a float representing the probability that the image is REAL.
    """
    if model is None:
        return 0.5 # Safe fallback if model isn't trained yet
        
    try:
        # Resize to 224x224 and convert to RGB
        img = np.array(img_pil.convert("RGB").resize((224, 224))) / 255.0
        inp = img[np.newaxis].astype(np.float32)
        
        # Predict
        out = model.predict(inp, verbose=0)
        
        # Binary classification: out[0][0] is the probability of class 1 (REAL)
        # CIFAKE structure: fake = 0, real = 1
        return float(out[0][0])
    except Exception as e:
        print(f"Error in Generalized Model: {e}")
        return 0.5
