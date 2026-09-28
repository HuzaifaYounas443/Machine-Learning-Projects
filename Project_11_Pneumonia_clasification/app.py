import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medical-ai-secret'
app.config['UPLOAD_FOLDER'] = 'static\\uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the model relative to this file so startup does not depend on the shell's cwd.
MODEL_PATH = Path(__file__).resolve().parent / 'Pneumonia detection training files' / 'pneumonia_model.keras'
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as error:
    model = None
    print(f"Model failed to load from {MODEL_PATH}: {error}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_pneumonia(image_path):
    """Predict pneumonia from image"""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)
    probability = float(prediction[0][0])
    
    # Return result
    if probability > 0.5:
        return "PNEUMONIA", probability * 100
    else:
        return "NORMAL", (1 - probability) * 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Model not loaded', 'error')
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Predict
        result, confidence = predict_pneumonia(filepath)
        
        return render_template('result.html', 
                             result=result,
                             confidence=confidence,
                             image_path=f"uploads/{unique_filename}")
    
    flash('Invalid file type', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)