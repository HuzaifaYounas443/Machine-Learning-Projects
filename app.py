# app/app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Boston House Value Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('models/boston_model.pkl')

model = load_model()

st.title("🏠 Boston House Value Prediction")
st.markdown("Predict median house value based on **average number of rooms** using linear regression.")

# ------------------------------------------------------------
# 1. CREATE A "TABLE-LIKE" INPUT SECTION
# ------------------------------------------------------------
st.subheader("📊 Enter Feature Values")

# Use columns to create a horizontal row (like a table row)
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust width ratios

with col1:
    st.write("**Feature**")  # Header
    st.write("Rooms (rm)")

with col2:
    st.write("**Value**")    # Header
    # Number input with WHOLE NUMBERS ONLY (step=1)
    rooms = st.number_input(
        " ",  # Empty label to keep it clean
        min_value=4,
        max_value=10,
        value=6,
        step=1,
        key="rooms_input"
    )

with col3:
    st.write("**Range**")    # Header
    st.write("1 - 10")

# Add a separator for visual clarity
st.divider()

# ------------------------------------------------------------
# 2. MAKE PREDICTION
# ------------------------------------------------------------
# Note: The model was trained on continuous data (e.g., 6.575). 
# If you input whole numbers (e.g., 6), it will still predict perfectly fine.
predicted_price = model.predict([[rooms]])[0]

st.subheader(" Prediction Result")
st.metric(label="Predicted Median House Value", value=f"${predicted_price * 1000:,.0f}")

# ------------------------------------------------------------
# 3. VISUALIZATION (Optional)
# ------------------------------------------------------------
st.subheader("📈 Visualization")

try:
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    X = df[['rm']]
    y = df['medv']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color='blue', alpha=0.5, label='Data points')
    
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    ax.plot(x_range, y_range, color='red', linewidth=2, label='Regression line')
    
    # Highlight user's input
    ax.scatter([rooms], [predicted_price], color='green', s=120, zorder=5, label='Your prediction')
    
    ax.set_xlabel('Average number of rooms')
    ax.set_ylabel('Median house value ($1000s)')
    ax.set_title('Rooms vs Value with Regression Line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
except Exception as e:
    st.warning("Could not load dataset for visualization. Only prediction is available.")

st.caption("Model trained on Boston Housing dataset. Enter whole numbers between 1 and 10.")