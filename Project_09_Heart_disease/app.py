# app.py
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

# Feature names in the order the model expects
FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

def load_model_and_scaler():
    """Load the trained Keras model and the fitted StandardScaler."""
    global model, scaler

    # Paths to saved artifacts (adjust if needed)
    model_path = os.path.join(os.path.dirname(__file__), 'diabetes_model_final.keras')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

    # If files don't exist in current dir, try parent (for deployment)
    if not os.path.exists(model_path):
        model_path = 'diabetes_model_final.keras'
    if not os.path.exists(scaler_path):
        scaler_path = 'scaler.pkl'

    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully from:", model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    try:
        scaler = joblib.load(scaler_path)
        print(" Scaler loaded successfully from:", scaler_path)
    except Exception as e:
        print(f" Error loading scaler: {e}")
        scaler = None

    return model is not None and scaler is not None

@app.route('/', methods=['GET'])
def index():
    """Render the main page with no prediction yet."""
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission, run prediction, and return result."""
    if model is None or scaler is None:
        return render_template('index.html', 
                               prediction=None, 
                               error="Model or scaler not loaded. Please check server logs.")

    try:
        # 1. Extract form data
        input_data = []
        for feature in FEATURES:
            val = request.form.get(feature)
            if val is None or val == '':
                return render_template('index.html', 
                                       prediction=None, 
                                       error=f"Missing value for: {feature}")
            input_data.append(float(val))

        # 2. Convert to numpy array and reshape for scaler
        input_array = np.array(input_data).reshape(1, -1)

        # 3. Scale using the fitted scaler
        scaled_input = scaler.transform(input_array)

        # 4. Predict (model outputs probability for class 1 = disease)
        prediction_prob = model.predict(scaled_input, verbose=0)[0][0]

        # 5. Render template with prediction
        return render_template('index.html', prediction=float(prediction_prob), error=None)

    except ValueError as e:
        return render_template('index.html', 
                               prediction=None, 
                               error=f"Invalid input: {str(e)}")
    except Exception as e:
        return render_template('index.html', 
                               prediction=None, 
                               error=f"Prediction error: {str(e)}")

# For local development
if __name__ == '__main__':
    # Load model and scaler before starting the server
    print("=" * 50)
    print("Loading Heart Disease Prediction Model...")
    success = load_model_and_scaler()
    if success:
        print(" Server ready! Navigate to http://127.0.0.1:5000")
    else:
        print(" Server starting but model/scaler not loaded. Predictions will fail.")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)