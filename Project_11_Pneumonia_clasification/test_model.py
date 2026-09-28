# test_model.py - Run this to verify model works
import numpy as np
import tensorflow as tf
from PIL import Image
import os

def test_model():
    model_path = 'Pneumonia detection training files/pneumonia_model.keras'
    
    print("Testing model...")
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded")
    
    # Test 1: Random data
    print("\nTest 1: Random data")
    random_input = np.random.rand(1, 224, 224, 3)
    pred = model.predict(random_input, verbose=0)
    print(f"  Prediction: {pred[0][0]:.6f}")
    print(f"  Expected: ~0.5")
    
    # Test 2: All zeros
    print("\nTest 2: All zeros")
    zero_input = np.zeros((1, 224, 224, 3))
    pred = model.predict(zero_input, verbose=0)
    print(f"  Prediction: {pred[0][0]:.6f}")
    
    # Test 3: All ones
    print("\nTest 3: All ones")
    ones_input = np.ones((1, 224, 224, 3))
    pred = model.predict(ones_input, verbose=0)
    print(f"  Prediction: {pred[0][0]:.6f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    
    if pred[0][0] > 0.99:
        print("⚠️ Model always predicts PNEUMONIA")
        print("  The model file might be corrupted")
        print("  Retrain the model or check file")
    elif pred[0][0] < 0.01:
        print("⚠️ Model always predicts NORMAL")
        print("  The model file might be corrupted")
    else:
        print("✓ Model seems to be working!")
    
    return model

if __name__ == "__main__":
    test_model()