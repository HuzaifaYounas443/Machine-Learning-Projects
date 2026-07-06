# app/app.py (Project 4 - Logistic Regression Classifier)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")

# --------------------------
# Load Model, Scaler, and Metadata
# --------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('Project_04_Breast_Cancer_Classification/models/logistic_model.pkl')
    scaler = joblib.load('Project_04_Breast_Cancer_Classification/models/scaler.pkl')
    metadata = joblib.load('Project_04_Breast_Cancer_Classification/models/metadata.pkl')
    return model, scaler, metadata

model, scaler, metadata = load_artifacts()
feature_names = metadata['feature_names']
target_names = metadata['target_names']  # ['malignant', 'benign']

st.title("🔬 Breast Cancer Prediction")
st.markdown("**Logistic Regression Classifier**")
st.caption("Enter the cell nucleus measurements below to predict if the tumor is **Malignant** or **Benign**.")
st.divider()

# --------------------------
# 1. Dynamic Input Grid (All 30 Features)
# --------------------------
st.subheader(" Enter Cell Measurements")
st.caption("The model was trained on standardized data. We will scale your inputs automatically.")

# We'll arrange inputs in 3 columns to save vertical space
cols = st.columns(3)
user_inputs = {}

for i, feature in enumerate(feature_names):
    col = cols[i % 3]  # Distribute across 3 columns
    with col:
        # Get min/max from the dataset to provide sensible default ranges
        # We'll hardcode reasonable ranges based on the dataset (mean ~10-30)
        if 'radius' in feature:
            default_val, min_val, max_val = 14.0, 5.0, 30.0
        elif 'texture' in feature:
            default_val, min_val, max_val = 19.0, 5.0, 40.0
        elif 'smoothness' in feature:
            default_val, min_val, max_val = 0.1, 0.01, 0.2
        elif 'compactness' in feature:
            default_val, min_val, max_val = 0.1, 0.01, 0.3
        elif 'concavity' in feature:
            default_val, min_val, max_val = 0.08, 0.0, 0.4
        elif 'concave points' in feature:
            default_val, min_val, max_val = 0.05, 0.0, 0.2
        elif 'symmetry' in feature:
            default_val, min_val, max_val = 0.18, 0.05, 0.3
        elif 'fractal_dimension' in feature:
            default_val, min_val, max_val = 0.06, 0.01, 0.1
        else:  # Fallback for 'worst' features
            default_val, min_val, max_val = 1.0, 0.0, 10.0
            
        user_inputs[feature] = st.number_input(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.01,
            key=f"input_{i}",
            format="%.4f"
        )

st.divider()

# --------------------------
# 2. Prediction
# --------------------------
# Convert dict to array in the correct order
input_array = np.array([user_inputs[f] for f in feature_names]).reshape(1, -1)

# --- IMPORTANT: Scale the input using the SAME scaler used in training ---
input_scaled = scaler.transform(input_array)

# Predict probability
pred_proba = model.predict_proba(input_scaled)[0]  # [prob_malignant, prob_benign]
pred_class = model.predict(input_scaled)[0]

# --------------------------
# 3. Display Results
# --------------------------
st.subheader(" Diagnosis Result")

# Create two columns for probability display
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Probability of Benign",
        value=f"{pred_proba[1] * 100:.1f}%",
        delta="Good" if pred_proba[1] > 0.5 else "Alert"
    )

with col2:
    st.metric(
        label="Probability of Malignant",
        value=f"{pred_proba[0] * 100:.1f}%",
        delta="Alert" if pred_proba[0] > 0.5 else "Good"
    )

# Diagnosis Box
if pred_class == 1:
    st.success(f" Prediction: **BENIGN** (Non-cancerous) with {pred_proba[1] * 100:.1f}% confidence.")
else:
    st.error(f" Prediction: **MALIGNANT** (Cancerous) with {pred_proba[0] * 100:.1f}% confidence.")

# --------------------------
# 4. Visual: Probability Gauge (Bar Chart)
# --------------------------
st.subheader(" Confidence Gauge")
fig, ax = plt.subplots(figsize=(6, 1.5))
ax.barh(['Probability'], [pred_proba[1]], color='green', alpha=0.7, label='Benign')
ax.barh(['Probability'], [pred_proba[0]], left=[pred_proba[1]], color='red', alpha=0.7, label='Malignant')
ax.set_xlim(0, 1)
ax.set_xlabel('Probability')
ax.legend(loc='upper right')
ax.set_title('Model Confidence Breakdown')
st.pyplot(fig)

# --------------------------
# 5. Feature Importance (Top Influential Features)
# --------------------------
with st.expander(" Top Features Influencing this Prediction"):
    # Calculate feature contributions for this specific input
    # Coef * scaled_value gives the log-odds contribution
    coefs = model.coef_[0]
    contribution = coefs * input_scaled.flatten()
    
    # Show top 5 positive and top 5 negative
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': contribution
    }).sort_values('Contribution', ascending=False)
    
    top_positive = contrib_df.head(5)
    top_negative = contrib_df.tail(5)
    
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.write("** Increases Benign Chance**")
        for _, row in top_positive.iterrows():
            st.write(f" {row['Feature']}: +{row['Contribution']:.3f}")
    with col_neg:
        st.write("** Increases Malignant Chance**")
        for _, row in top_negative.iterrows():
            st.write(f" {row['Feature']}: {row['Contribution']:.3f}")

st.divider()
st.caption("Model trained on Wisconsin Breast Cancer Dataset. Inputs are automatically standardized using the training scaler.")