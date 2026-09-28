import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

print("First 5 rows:")
print(df.head())

features = ['rm', 'lstat']
x = df[features]
y = df['medv']

print(f"Selected Features {features}")

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

degree = 3
alpha = 0.5

model = Pipeline([
    ('Poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('Scaler', StandardScaler()),
    ('Ridge', Ridge(alpha=alpha))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(df['rm'], df['medv'], alpha=0.4, color='blue', label='Actual Data')

rm_range = np.linspace(df['rm'].min(), df['rm'].max(), 100).reshape(-1, 1)
lstat_median = np.full((100, 1), df['lstat'].median())
x_curve = np.hstack([rm_range, lstat_median])

y_curve = model.predict(x_curve)

plt.plot(rm_range, y_curve, color='red', linewidth=3, label=f'Polynomial Degree {degree} (Ridge)')
plt.xlabel('Average Number of Rooms (rm)')
plt.ylabel('Median Value of Homes ($1000s)')
plt.title('Polynomial Regression with Ridge Regularization')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('polynomial_curve.png')
plt.show()

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/boston_polynomial_model.pkl')

metadata = {
    'features' : features,
    'degree' : degree, 
    'alpha' : alpha,
}

joblib.dump(metadata, 'models/model_metadata.pkl')
print('Model saved.')
print('Metadata saved.')