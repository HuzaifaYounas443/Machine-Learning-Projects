# app/app.py (Titanic Survival Prediction - COMPLETE & FIXED)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

# --------------------------
# Load Model and Metadata
# --------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('Project_06_Titanic_survival_prediction/models/titanic_model.pkl')
    metadata = joblib.load('Project_06_Titanic_survival_prediction/models/metadata.pkl')
    return model, metadata

model, metadata = load_artifacts()
feature_names = metadata['feature_names']
target_names = metadata['target_names']  # ['Died', 'Survived']

st.title(" Titanic Survival Predictor")
st.markdown("**Will you survive the Titanic disaster?**")
st.caption("Enter your passenger details below to find out!")

st.divider()

# --------------------------
# Input Section
# --------------------------
st.subheader(" Enter Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], 
                          help="1st = Upper, 2nd = Middle, 3rd = Lower")
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.slider("Age (years)", 0, 80, 30)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)

with col2:
    parch = st.number_input("Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = st.number_input("Ticket Fare ($)", 0, 600, 32)
    embarked = st.selectbox("Port of Embarkation", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])

st.divider()

# --------------------------
# Preprocess Input (DYNAMIC - HANDLES ANY COLUMN ORDER)
# --------------------------
# Convert sex to 0 (male) or 1 (female)
sex_value = 1 if sex == "Female" else 0

# Create a dictionary that maps user inputs to the exact feature names the model expects
input_dict = {}

for feature in feature_names:
    if feature == 'PassengerId':
        # Model expects PassengerId (dummy value)
        input_dict[feature] = 0
    elif feature == 'Pclass':
        input_dict[feature] = pclass
    elif feature == 'Sex':
        input_dict[feature] = sex_value
    elif feature == 'Age':
        input_dict[feature] = age
    elif feature == 'SibSp':
        input_dict[feature] = sibsp
    elif feature == 'Parch':
        input_dict[feature] = parch
    elif feature == 'Fare':
        input_dict[feature] = fare
    elif feature == 'Embarked_C':
        input_dict[feature] = 1 if "C" in embarked else 0
    elif feature == 'Embarked_Q':
        input_dict[feature] = 1 if "Q" in embarked else 0
    elif feature == 'Embarked_S':
        input_dict[feature] = 1 if "S" in embarked else 0
    else:
        # Fallback for any unknown feature (should not happen)
        input_dict[feature] = 0

# Convert dictionary to DataFrame (single row)
input_data = pd.DataFrame([input_dict])

# Ensure columns are in the EXACT SAME ORDER as training
input_data = input_data[feature_names]

# --------------------------
# Prediction
# --------------------------
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0]

# --------------------------
# Display Results
# --------------------------
st.subheader(" Prediction Result")

if prediction == 1:
    st.success(f" You would have SURVIVED the Titanic disaster!")
    st.metric("Survival Probability", f"{probability[1] * 100:.1f}%")
else:
    st.error(f" You would have PERISHED in the Titanic disaster.")
    st.metric("Survival Probability", f"{probability[1] * 100:.1f}%")

# --------------------------
# Probability Breakdown (Bar Chart)
# --------------------------
st.subheader(" Survival Chance Breakdown")

fig, ax = plt.subplots(figsize=(6, 2))

# Create a horizontal bar chart
ax.barh(['Probability'], [probability[0]], color='red', alpha=0.7, label='Perish')
ax.barh(['Probability'], [probability[1]], left=[probability[0]], color='green', alpha=0.7, label='Survive')

ax.set_xlim(0, 1)
ax.set_xlabel('Probability')
ax.legend(loc='upper right')
ax.set_title('Model Confidence Breakdown')

st.pyplot(fig)

# --------------------------
# Feature Importance (Optional - Show Top Features)
# --------------------------
with st.expander(" What factors influenced this prediction?"):
    st.markdown("""
    **Top features that determine survival:**
    1. **Sex** – Women had a much higher survival rate.
    2. **Passenger Class** – 1st class passengers had better access to lifeboats.
    3. **Fare** – Higher fare passengers were more likely to survive.
    4. **Age** – Children and elderly were prioritized.
    """)
    st.caption("These insights come from the Random Forest model's feature importance analysis.")

st.divider()
st.caption("Model trained on Titanic dataset using Random Forest Classifier.")