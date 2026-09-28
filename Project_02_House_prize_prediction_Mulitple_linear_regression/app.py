import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Boston Housing (Multiple Regression)", layout="centered")

# --------------------------
# Load model and define features
# --------------------------
@st.cache_resource
def load_model():
    return joblib.load('House_prize_prediction_Mulitple_linear_regression_Project_02/models/boston_multiple_model.pkl')

model = load_model()

# Feature definitions with safe integer ranges (whole numbers only)
FEATURES = {
    'rm': {'label': 'Average Rooms', 'min': 4, 'max': 9, 'default': 6},
    'lstat': {'label': 'Lower Status (%)', 'min': 1, 'max': 38, 'default': 10},
    'ptratio': {'label': 'Pupil-Teacher Ratio', 'min': 12, 'max': 22, 'default': 18},
    'indus': {'label': 'Non-Retail Business Acres', 'min': 0, 'max': 28, 'default': 5}
}
feature_order = ['rm', 'lstat', 'ptratio', 'indus']

st.title("🏠 Boston House Price Predictor")
st.markdown("**Multiple Linear Regression** using 4 key features.")

st.divider()

# --------------------------
# Dynamic Input Table (whole numbers)
# --------------------------
st.subheader("Enter Feature Values")

col1, col2, col3 = st.columns([1.5, 2, 1])

with col1:
    st.write("**Feature**")
with col2:
    st.write("**Value (whole number)**")
with col3:
    st.write("**Safe Range**")

user_inputs = {}
for feature in feature_order:
    params = FEATURES[feature]
    with col1:
        st.write(params['label'])
    with col2:
        user_inputs[feature] = st.number_input(
            " ",  # empty label to keep table clean
            min_value=params['min'],
            max_value=params['max'],
            value=params['default'],
            step=1,
            key=f"input_{feature}"
        )
    with col3:
        st.write(f"{params['min']} – {params['max']}")

st.divider()

# --------------------------
# Prediction
# --------------------------
input_array = np.array([[user_inputs[f] for f in feature_order]])
predicted_price = model.predict(input_array)[0]

st.subheader(" Prediction Result")
st.metric(
    label="Predicted Median House Value",
    value=f"${predicted_price * 1000:,.0f}"
)

# --------------------------
# Feature Impact Bar Chart (with clean labels)
# --------------------------
st.subheader(" Feature Impact")

# Use the same label mapping as in training
label_map = {
    'rm': 'Avg Rooms',
    'lstat': 'Lower Status %',
    'ptratio': 'Pupil-Teacher Ratio',
    'indus': 'Non-Retail Acres'
}
display_names = [label_map[f] for f in feature_order]
coefs = model.coef_

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['green' if c > 0 else 'red' for c in coefs]
bars = ax.barh(display_names, coefs, color=colors, alpha=0.7)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Impact on Price ($1000s)')
ax.set_title('How Each Feature Affects House Price')

# Add coefficient values on bars with proper alignment
for bar, val in zip(bars, coefs):
    if val > 0:
        x_pos = val + 0.1
        ha = 'left'
    else:
        x_pos = val - 0.1
        ha = 'right'
    
    ax.text(
        x_pos,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.2f}',
        va='center',
        ha=ha,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6)
    )

plt.tight_layout()
st.pyplot(fig)

st.caption("""
 **Green** = More of this feature increases price.  
 **Red** = More of this feature decreases price.
""")

# --------------------------
# Sidebar: Quick Comparison
# --------------------------
with st.sidebar:
    st.header(" Model Comparison")
    st.metric("Simple Linear (1 feature)", "R² = 0.37")
    st.metric("Multiple Linear (4 features)", "R² ≈ 0.72", delta="+0.35")
    st.caption("Better features + avoiding multicollinearity → stronger model.")
    
    st.divider()
    
    st.header(" Model Equation")
    equation = f"Price = {model.intercept_:.2f}"
    for name, coef in zip(feature_order, model.coef_):
        equation += f" + ({coef:.2f} × {FEATURES[name]['label']})"
    st.code(equation, language="text")
    
    st.divider()
    
    st.header(" Feature Importance Ranking")
    importance_df = pd.DataFrame({
        'Feature': display_names,
        'Coefficient': coefs,
        'Abs Impact': np.abs(coefs)
    }).sort_values('Abs Impact', ascending=False)
    st.dataframe(importance_df, hide_index=True, use_container_width=True)

st.divider()
st.caption("Model trained on Boston Housing dataset. Inputs are restricted to the original data range to avoid extrapolation.")