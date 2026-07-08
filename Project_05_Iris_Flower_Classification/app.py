# app/app.py (Project 5 - Iris Flower Classification)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Iris Flower Classifier", layout="centered")

# --------------------------
# Load Models and Metadata
# --------------------------
@st.cache_resource
def load_artifacts():
    dt_model = joblib.load('Project_05_Iris_Flower_Classification/models/decision_tree.pkl')
    rf_model = joblib.load('Project_05_Iris_Flower_Classification/models/random_forest.pkl')
    metadata = joblib.load('Project_05_Iris_Flower_Classification/models/metadata.pkl')
    return dt_model, rf_model, metadata

dt_model, rf_model, metadata = load_artifacts()
feature_names = metadata['feature_names']
target_names = metadata['target_names']

st.title("🌸 Iris Flower Classification")
st.markdown("**Decision Tree vs Random Forest**")
st.caption("Enter the flower measurements below to predict the species.")

st.divider()

# --------------------------
# Input Section
# --------------------------
st.subheader(" Enter Flower Measurements")

# Create 2 columns for inputs
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.8,
        step=0.1
    )
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=4.5,
        value=3.0,
        step=0.1
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=4.0,
        step=0.1
    )
    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=2.5,
        value=1.2,
        step=0.1
    )

st.divider()

# --------------------------
# Prediction
# --------------------------
input_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Decision Tree
pred_dt = dt_model.predict(input_array)[0]
proba_dt = dt_model.predict_proba(input_array)[0]

# Random Forest
pred_rf = rf_model.predict(input_array)[0]
proba_rf = rf_model.predict_proba(input_array)[0]

# --------------------------
# Display Results
# --------------------------
st.subheader(" Prediction Results")

# Create tabs for comparison
tab1, tab2 = st.tabs([" Random Forest (Recommended)", " Decision Tree"])

with tab1:
    st.success(f"**Predicted Species:** {target_names[pred_rf].upper()}")
    st.caption(f"Confidence: {proba_rf[pred_rf] * 100:.1f}%")
    
    # Show probability breakdown
    fig, ax = plt.subplots(figsize=(6, 2))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax.barh(['Probability'], [proba_rf[0]], color=colors[0], alpha=0.7, label=target_names[0])
    ax.barh(['Probability'], [proba_rf[1]], left=[proba_rf[0]], color=colors[1], alpha=0.7, label=target_names[1])
    ax.barh(['Probability'], [proba_rf[2]], left=[proba_rf[0] + proba_rf[1]], color=colors[2], alpha=0.7, label=target_names[2])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.legend(loc='lower right')
    st.pyplot(fig)

with tab2:
    st.info(f"**Predicted Species:** {target_names[pred_dt].upper()}")
    st.caption(f"Confidence: {proba_dt[pred_dt] * 100:.1f}%")
    
    # Show probability breakdown
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.barh(['Probability'], [proba_dt[0]], color=colors[0], alpha=0.7, label=target_names[0])
    ax.barh(['Probability'], [proba_dt[1]], left=[proba_dt[0]], color=colors[1], alpha=0.7, label=target_names[1])
    ax.barh(['Probability'], [proba_dt[2]], left=[proba_dt[0] + proba_dt[1]], color=colors[2], alpha=0.7, label=target_names[2])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.legend(loc='lower right')
    st.pyplot(fig)

# --------------------------
# Feature Importance (Random Forest)
# --------------------------
st.subheader(" Feature Importance (Random Forest)")

# Load feature importance from trained model
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
ax.set_xlabel('Importance Score')
ax.set_title('Which Features Matter Most?')
ax.grid(axis='x', alpha=0.3)
st.pyplot(fig)

st.caption("""
**Petal Length** and **Petal Width** are the most important features for classifying Iris species.
""")

# --------------------------
# Model Comparison
# --------------------------
with st.sidebar:
    st.header(" Model Comparison")
    st.markdown("""
    | Feature | Decision Tree | Random Forest |
    | :--- | :--- | :--- |
    | **Type** | Single Tree | Ensemble (100 Trees) |
    | **Overfitting** | Higher risk | Lower risk |
    | **Interpretability** | Easy to visualize | Harder to interpret |
    | **Speed** | Faster | Slower |
    | **Accuracy** | Good | Better |
    """)
    st.caption("**Random Forest** is usually the better choice for real-world applications.")

st.divider()
st.caption("Model trained on Iris Dataset using scikit-learn.")