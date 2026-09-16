# app/utils.py
import re
import string
import joblib
import os


def _ensure_model_compatibility(model):
    """Fix scikit-learn compatibility gaps in older pickled SVC models."""
    if model is None:
        return model

    if not hasattr(model, '_effective_probability') and getattr(model, 'probability', False):
        model._effective_probability = True

    return model


def clean_text(text):
    """Clean and preprocess text for prediction"""
    if not text:
        return text
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

def load_models():
    """Load the trained model and vectorizer"""
    model_path = 'Project_08_SPAM_Email_Detector/models/spam_model.pkl'
    vectorizer_path = 'Project_08_SPAM_Email_Detector/models/vectorizer.pkl'
    metadata_path = 'Project_08_SPAM_Email_Detector/models/metadata.pkl'
    
    if not all(os.path.exists(p) for p in [model_path, vectorizer_path, metadata_path]):
        return None, None, None
    
    model = joblib.load(model_path)
    model = _ensure_model_compatibility(model)
    vectorizer = joblib.load(vectorizer_path)
    metadata = joblib.load(metadata_path)
    
    return model, vectorizer, metadata

def predict_message(message, model, vectorizer):
    """Predict if a message is spam or ham"""
    if not message or model is None or vectorizer is None:
        return None, None
    
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0][1]
    
    return prediction, probability

def get_spam_indicators(text):
    """Get spam indicators from a message"""
    spam_keywords = [
        'free', 'win', 'winner', 'cash', 'prize', 'click', 'subscribe',
        'urgent', 'limited', 'offer', 'exclusive', 'guaranteed',
        'congratulations', 'claim', 'bonus', 'credit', 'money',
        'million', 'dollar', 'iphone', 'amazon', 'gift', 'card'
    ]
    
    found = [word for word in spam_keywords if word in text.lower()]
    return found