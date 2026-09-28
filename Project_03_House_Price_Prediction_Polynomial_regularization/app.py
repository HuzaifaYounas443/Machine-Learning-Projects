# app/app.py (Project 3 - Polynomial Regression)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Boston Housing (Polynomial Regression)", layout="centered")

# --------------------------
# Load model and metadata
# --------------------------
@st.cache_resource
def load_model():
    model = joblib.load('Project_03/models/boston_polynomial_model.pkl')
    metadata = joblib.load('Project_03/models/model_metadata.pkl')
    return model, metadata

model, metadata = load_model()
features = metadata['features']  # ['rm', 'lstat']
degree = metadata['degree']
alpha = metadata['alpha']

st.title(" Polynomial Regression Explorer")
st.markdown("**Project 3:** Capturing non-linear relationships using Polynomial Features + Ridge Regularization.")
st.caption(f"Current model: Degree {degree}, Ridge Alpha = {alpha}")

st.divider()

# --------------------------
# 1. Input Section (Same clean table as Project 2)
# --------------------------
st.subheader(" Enter Feature Values")

FEATURES = {
    'rm': {'label': 'Average Rooms', 'min': 4, 'max': 9, 'default': 6},
    'lstat': {'label': 'Lower Status (%)', 'min': 1, 'max': 38, 'default': 10}
}
feature_order = ['rm', 'lstat']

col1, col2, col3 = st.columns([1.5, 2, 1])

with col1:
    st.write("**Feature**")
with col2:
    st.write("**Value**")
with col3:
    st.write("**Safe Range**")

user_inputs = {}
for feature in feature_order:
    params = FEATURES[feature]
    with col1:
        st.write(params['label'])
    with col2:
        user_inputs[feature] = st.number_input(
            " ",
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
# 2. Prediction
# --------------------------
input_array = np.array([[user_inputs[f] for f in feature_order]])
predicted_price = model.predict(input_array)[0]

st.subheader(" Prediction Result")
st.metric(
    label="Predicted Median House Value",
    value=f"${predicted_price * 1000:,.0f}"
)

# --------------------------
# 3. VISUALIZATION: Show the Polynomial Curve
# --------------------------
st.subheader("Polynomial Fit Visualization")
st.caption("The red curve shows the non-linear relationship between rooms and price (holding LSTAT fixed).")
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
# Create the curve plot
rm_range = np.linspace(4, 9, 100).reshape(-1, 1)
lstat_median = np.full((100, 1), df['lstat'].median())  # Need to load df
# We need to load the dataset just for plotting


X_curve = np.hstack([rm_range, lstat_median])
y_curve = model.predict(X_curve)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df['rm'], df['medv'], alpha=0.3, color='blue', label='Actual Data')
ax.plot(rm_range, y_curve, color='red', linewidth=3, label='Polynomial Curve (Ridge)')
# Highlight user input
ax.scatter([user_inputs['rm']], [predicted_price], color='green', s=100, zorder=5, label='Your Prediction')
ax.set_xlabel('Average Rooms (rm)')
ax.set_ylabel('Median Value ($1000s)')
ax.set_title(f'Polynomial Degree {degree} (Holding LSTAT at median)')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# --------------------------
# 4. Explain the Magic
# --------------------------
with st.expander(" Why is this better than simple linear regression?"):
    st.markdown("""
    - **Linear (Project 1):** Assumes price increases at the same rate forever (straight line). 
    - **Polynomial (Project 3):** Allows the curve to bend. Notice how the red curve flattens out at high room counts? That's because extremely large houses don't scale perfectly linearly in price.
    - **Ridge Regularization:** Stops the curve from going crazy (overfitting). Without it, a degree-3 polynomial would wiggle wildly to hit every single data point. Ridge smooths it out so it works well on new houses.
    """)

st.divider()
st.caption("Model trained using PolynomialFeatures (degree 3) + Ridge Regression on Boston Housing.")