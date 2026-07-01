# train_model.py (Project 2 - Multiple Linear Regression)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# --------------------------
# 1. Load dataset
# --------------------------
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

print("First 5 rows:")
print(df.head())

# --------------------------
# 2. Feature Selection (Avoiding Multicollinearity)
# --------------------------
# We pick 4 features with strong correlations to 'medv' but low correlation among themselves.
# 'tax' and 'rad' are dropped because they correlate ~0.91 with each other.
features = ['rm', 'lstat', 'ptratio', 'indus']
X = df[features]
y = df['medv']

print(f"\nSelected Features: {features}")
print("\nCorrelation with target:")
print(df[features + ['medv']].corr()['medv'].sort_values(ascending=False))

# --------------------------
# 3. Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# --------------------------
# 4. Train Model
# --------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Display coefficients
print("\n--- Model Coefficients ---")
for feature, coef in zip(features, model.coef_):
    print(f"{feature:>10}: {coef:>10.4f}")
print(f"{'Intercept':>10}: {model.intercept_:>10.4f}")

# --------------------------
# 5. Evaluate
# --------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"MAE:  {mae:.2f} ($1000s)")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f} ($1000s)")
print(f"R²:   {r2:.4f}")

# --------------------------
# 6. Feature Importance Visualization (with clean labels)
# --------------------------
plt.figure(figsize=(8, 5))

# Human‑readable labels to avoid confusion (e.g., lstat vs Istat)
label_map = {
    'rm': 'Average Rooms (rm)',
    'lstat': 'Lower Status % (lstat)',
    'ptratio': 'Pupil-Teacher Ratio',
    'indus': 'Non-Retail Business Acres'
}

coef_df = pd.DataFrame({
    'Feature': [label_map[f] for f in features],
    'Coefficient': model.coef_
})
coef_df = coef_df.sort_values('Coefficient', ascending=False)

# Plot bars
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='steelblue', alpha=0.8)
plt.xlabel('Coefficient Value (impact on price in $1000s)')
plt.title('Feature Impact on House Price (Multiple Regression)')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(axis='x', alpha=0.3)

# Add value labels on bars
for index, value in enumerate(coef_df['Coefficient']):
    plt.text(value + 0.3 if value > 0 else value - 1.5,
             index,
             f'{value:.2f}',
             va='center',
             fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# --------------------------
# 7. Save Model
# --------------------------
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/boston_multiple_model.pkl')
print("\n✅ Model saved to 'models/boston_multiple_model.pkl'")