# app.py - Streamlit Frontend for Pneumonia Detection
# Place this file in your project directory

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Pneumonia Detection from Chest X-Rays",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1a5276;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .normal {
        background: linear-gradient(135deg, #d5f5e3, #a9dfbf);
        border: 2px solid #27ae60;
    }
    .pneumonia {
        background: linear-gradient(135deg, #fadbd8, #f1948a);
        border: 2px solid #e74c3c;
    }
    .confidence-bar {
        height: 35px;
        border-radius: 8px;
        margin: 10px 0;
        transition: width 0.5s;
        background: linear-gradient(90deg, #2ecc71, #f1c40f, #e74c3c);
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        color: white;
        font-weight: 600;
        height: 55px;
        font-size: 1.1rem;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 82, 118, 0.4);
    }
    .upload-area {
        border: 2px dashed #1a5276;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        background: #f8f9fa;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 20px;
        margin-top: 30px;
        border-top: 1px solid #ecf0f1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<p class="main-header">🫁 Pneumonia Detection from Chest X-Rays</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a chest X-ray image for instant AI-powered pneumonia detection</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://www.who.int/images/default-source/wpro/health-topics/pneumonia/pneumonia-infographic.tmb-1920v.jpg?Culture=en&sfvrsn=15b84684_2", 
             caption="About Pneumonia", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "89.6%", delta="0.4%")
        st.metric("Precision", "93.8%", delta="0.2%")
    with col2:
        st.metric("Recall", "89.2%", delta="0.3%")
        st.metric("F1-Score", "91.5%", delta="0.3%")
    
    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    1. **Upload** a chest X-ray image
    2. **Analyze** using our VGG16-based deep learning model
    3. **Get Results** with confidence score
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    This tool is for **educational and research purposes only**. 
    Not for clinical use. Always consult healthcare professionals.
    """)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    # Try multiple possible paths
    possible_paths = [
        '/kaggle/working/pneumonia_model.keras',  # Kaggle path
        'pneumonia_model.keras',                   # Local path
        './pneumonia_model.keras',                # Current directory
        '../pneumonia_model.keras',               # Parent directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                st.success(f"✅ Model loaded successfully from {path}")
                return model
            except Exception as e:
                st.warning(f"⚠️ Failed to load from {path}: {str(e)}")
                continue
    
    # If no model found, show error and allow upload
    st.error("❌ Model not found. Please upload your model file.")
    uploaded_model = st.file_uploader(
        "Upload model file (.keras)",
        type=['keras'],
        help="Upload the pneumonia_model.keras file"
    )
    
    if uploaded_model is not None:
        try:
            # Save uploaded model
            with open('uploaded_model.keras', 'wb') as f:
                f.write(uploaded_model.getbuffer())
            model = tf.keras.models.load_model('uploaded_model.keras')
            st.success("✅ Model loaded successfully from upload!")
            return model
        except Exception as e:
            st.error(f"❌ Error loading uploaded model: {str(e)}")
            return None
    
    return None

# ============================================
# IMAGE PREPROCESSING
# ============================================
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def apply_clahe(image):
    """Apply CLAHE enhancement to image"""
    try:
        # Convert PIL to OpenCV
        img_cv = np.array(image)
        
        # Handle grayscale images
        if len(img_cv.shape) == 2:
            gray = img_cv
        else:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(enhanced_rgb)
    except Exception as e:
        st.warning(f"CLAHE enhancement failed: {str(e)}")
        return image

# ============================================
# MAIN CONTENT
# ============================================
# Load model
model = load_model()

# Create columns for layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload X-Ray Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a chest X-ray image in JPG, JPEG, PNG, BMP, or TIFF format"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        apply_clahe_option = st.checkbox(
            "Apply CLAHE Enhancement", 
            value=False, 
            help="Enhances image contrast for potentially better detection"
        )
        show_probabilities = st.checkbox(
            "Show detailed probabilities",
            value=True,
            help="Show probability scores for both classes"
        )
    
    # Quick example images (optional)
    st.markdown("#### 📷 Quick Test")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Test with Normal X-Ray"):
            st.session_state['test_image'] = 'normal'
    with col_b:
        if st.button("Test with Pneumonia X-Ray"):
            st.session_state['test_image'] = 'pneumonia'

with col2:
    st.markdown("### 🖼️ Image Preview")
    
    # Display image
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Apply CLAHE if selected
        if apply_clahe_option:
            image_processed = apply_clahe(image)
            st.image(image_processed, caption="Enhanced Image (CLAHE Applied)", use_container_width=True)
        else:
            st.image(image, caption="Original Image", use_container_width=True)
        
        # Store image for prediction
        st.session_state['image'] = image
        st.session_state['processed_image'] = image_processed if apply_clahe_option else image
        st.session_state['uploaded'] = True
    else:
        st.info("📷 Upload an image to see preview here")
        # Placeholder image
        placeholder = Image.new('RGB', (400, 400), color='#f0f0f0')
        st.image(placeholder, caption="No image uploaded", use_container_width=True)
        st.session_state['uploaded'] = False

# ============================================
# PREDICTION BUTTON
# ============================================
st.markdown("---")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_button = st.button("🔬 Analyze Image for Pneumonia", use_container_width=True)

# ============================================
# PREDICTION LOGIC
# ============================================
if predict_button:
    if not st.session_state.get('uploaded', False):
        st.error("⚠️ Please upload an image first!")
    elif model is None:
        st.error("⚠️ Model not loaded. Please ensure model file exists.")
    else:
        with st.spinner("🔄 Analyzing image... This may take a few seconds."):
            try:
                # Get the image
                image = st.session_state['processed_image']
                
                # Preprocess
                preprocessed = preprocess_image(image)
                
                if preprocessed is None:
                    st.error("❌ Failed to preprocess image")
                else:
                    # Make prediction
                    prediction = model.predict(preprocessed, verbose=0)
                    confidence = float(prediction[0][0])
                    
                    # Determine class
                    if confidence > 0.5:
                        predicted_class = "PNEUMONIA"
                        confidence_percentage = confidence * 100
                        color = "#e74c3c"
                        icon = "⚠️"
                    else:
                        predicted_class = "NORMAL"
                        confidence_percentage = (1 - confidence) * 100
                        color = "#27ae60"
                        icon = "✅"
                    
                    # Store results
                    st.session_state['prediction'] = predicted_class
                    st.session_state['confidence'] = confidence_percentage
                    st.session_state['raw_prediction'] = confidence
                    st.session_state['color'] = color
                    st.session_state['icon'] = icon
                    
                    # Success message
                    st.success("✅ Analysis complete!")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

# ============================================
# DISPLAY RESULTS
# ============================================
if 'prediction' in st.session_state:
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")
    
    # Create result columns
    result_col1, result_col2 = st.columns([3, 2])
    
    with result_col1:
        # Prediction box
        color = st.session_state['color']
        icon = st.session_state['icon']
        prediction = st.session_state['prediction']
        confidence = st.session_state['confidence']
        
        st.markdown(f"""
        <div class="prediction-box" style="background: {color}22; border: 2px solid {color};">
            <h2 style="color: {color};">{icon} {prediction}</h2>
            <p style="font-size: 1.2rem; color: #2c3e50;">
                {'Pneumonia detected' if prediction == 'PNEUMONIA' else 'No signs of pneumonia detected'}
            </p>
            <p style="font-size: 1.1rem; color: #2c3e50;">
                Confidence: <strong style="color: {color};">{confidence:.2f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col2:
        # Confidence gauge
        st.markdown("#### Confidence Score")
        
        # Create a gauge chart
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh([''], [confidence], color=color, height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Confidence (%)')
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.grid(axis='x', alpha=0.3)
        ax.set_title(f'Confidence: {confidence:.1f}%')
        st.pyplot(fig)
        plt.close()
    
    # ============================================
    # DETAILED METRICS
    # ============================================
    st.markdown("---")
    st.markdown("#### 📈 Detailed Analysis")
    
    # Create metrics row
    metric_cols = st.columns(4)
    raw_prob = st.session_state['raw_prediction']
    
    with metric_cols[0]:
        st.metric("Predicted Class", prediction, delta="AI Decision")
    
    with metric_cols[1]:
        st.metric("Confidence", f"{confidence:.1f}%", 
                 delta="High" if confidence > 80 else "Medium" if confidence > 60 else "Low")
    
    with metric_cols[2]:
        normal_prob = (1 - raw_prob) * 100
        st.metric("Normal Probability", f"{normal_prob:.1f}%")
    
    with metric_cols[3]:
        pneumonia_prob = raw_prob * 100
        st.metric("Pneumonia Probability", f"{pneumonia_prob:.1f}%")
    
    # ============================================
    # PROBABILITY BARS
    # ============================================
    if show_probabilities:
        st.markdown("#### Class Probabilities")
        
        fig, ax = plt.subplots(figsize=(8, 3))
        classes = ['NORMAL', 'PNEUMONIA']
        probs = [(1 - raw_prob) * 100, raw_prob * 100]
        colors = ['#27ae60', '#e74c3c']
        
        bars = ax.bar(classes, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probability (%)')
        ax.set_title('Class Probabilities')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    # ============================================
    # INTERPRETATION GUIDE
    # ============================================
    with st.expander("📖 Understanding Your Results"):
        st.markdown("""
        ### Interpretation Guide
        
        | Result | Meaning | Action |
        |--------|---------|---------|
        | **NORMAL** | Model did not detect patterns consistent with pneumonia | Continue monitoring; consult doctor if symptoms persist |
        | **PNEUMONIA** | Model detected patterns consistent with pneumonia | Consult a healthcare professional immediately |
        
        ### Confidence Levels
        - **> 90%**: Very high confidence - Prediction is very reliable
        - **70-90%**: High confidence - Prediction is likely correct
        - **50-70%**: Moderate confidence - Consider additional testing
        - **< 50%**: Low confidence - Model is uncertain, consider re-evaluation
        
        ### Important Notes
        - This is a **screening tool**, not a diagnostic tool
        - Always consult with healthcare professionals
        - False positives and false negatives are possible
        - The model was trained on a specific dataset and may not generalize to all populations
        """)

# ============================================
# BATCH PROCESSING (Optional)
# ============================================
st.markdown("---")
with st.expander("🔄 Batch Processing (Advanced)"):
    st.markdown("""
    Upload multiple images for batch analysis. This is useful for processing multiple X-rays at once.
    """)
    
    batch_files = st.file_uploader(
        "Upload multiple images",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True
    )
    
    if batch_files and model is not None:
        if st.button("Process Batch"):
            with st.spinner("Processing batch..."):
                results = []
                for file in batch_files:
                    try:
                        image = Image.open(file)
                        preprocessed = preprocess_image(image)
                        if preprocessed is not None:
                            pred = model.predict(preprocessed, verbose=0)
                            prob = float(pred[0][0])
                            class_name = "PNEUMONIA" if prob > 0.5 else "NORMAL"
                            results.append({
                                'filename': file.name,
                                'prediction': class_name,
                                'confidence': prob * 100 if prob > 0.5 else (1 - prob) * 100
                            })
                    except Exception as e:
                        results.append({
                            'filename': file.name,
                            'prediction': 'ERROR',
                            'confidence': 0
                        })
                
                # Display results
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    ### 📚 References
    - [Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    - [WHO Pneumonia Info](https://www.who.int/health-topics/pneumonia)
    """)

with footer_col2:
    st.markdown("""
    ### 🔗 Links
    - [Report Issue](https://github.com/yourusername/pneumonia-detection/issues)
    - [Source Code](https://github.com/yourusername/pneumonia-detection)
    """)

with footer_col3:
    st.markdown(f"""
    ### 📅 Info
    - Version: 1.0.0
    - Model: VGG16
    - Updated: {datetime.now().strftime('%Y-%m-%d')}
    """)

st.markdown(f"""
<div class="footer">
    Made with ❤️ using TensorFlow and Streamlit • For educational purposes only
</div>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE CLEANUP
# ============================================
# Reset session state if needed
if st.sidebar.button("🔄 Reset App"):
    for key in ['prediction', 'confidence', 'raw_prediction', 'uploaded']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()