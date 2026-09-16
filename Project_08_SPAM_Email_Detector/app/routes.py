# app/routes.py
from flask import Blueprint, render_template, request, jsonify, session
from app.utils import load_models, predict_message, clean_text, get_spam_indicators

main = Blueprint('main', __name__)

# Load models once at startup
model, vectorizer, metadata = load_models()

@main.route('/', methods=['GET', 'POST'])
def index():
    """Home page with spam detection"""
    result = None
    error = None
    
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        
        if not message:
            error = "Please enter a message to check."
        else:
            prediction, probability = predict_message(message, model, vectorizer)
            
            if prediction is None:
                error = "Model not loaded. Please train the model first."
            else:
                # Get spam indicators
                indicators = get_spam_indicators(message)
                
                result = {
                    'message': message,
                    'is_spam': bool(prediction),
                    'probability': probability,
                    'indicators': indicators,
                    'cleaned_message': clean_text(message)
                }
    
    return render_template('index.html', result=result, error=error)

@main.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    prediction, probability = predict_message(message, model, vectorizer)
    
    if prediction is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'is_spam': bool(prediction),
        'probability': probability,
        'clean_message': clean_text(message),
        'indicators': get_spam_indicators(message)
    })

@main.route('/about')
def about():
    """About page"""
    return render_template('about.html', metadata=metadata)