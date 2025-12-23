# =========================================
# 0. Import Libraries
# =========================================
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# =========================================
# 1. Load Dataset
# =========================================
df = pd.read_csv("insurance.csv")

print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())


# =========================================
# 2. Encoding Categorical Features
# =========================================
categorical_cols = ['sex', 'smoker', 'region']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nEncoded Data Sample:")
print(df.head())


# =========================================
# 3. Exploratory Data Analysis (EDA)
# Scatter Plot: BMI vs Charges colored by Smoker
# =========================================
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='bmi',
    y='charges',
    hue=label_encoders['smoker'].inverse_transform(df['smoker'])
)
plt.title("BMI vs Charges (Colored by Smoker)")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.legend(title="Smoker")
plt.show()


# =========================================
# 4. Prepare Features and Target
# =========================================
X = df.drop('charges', axis=1)
y = df['charges']


# =========================================
# 5. Train / Test Split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# =========================================
# 6. Linear Regression (Degree = 1)
# =========================================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_train_pred_lr = lin_reg.predict(X_train)
y_test_pred_lr = lin_reg.predict(X_test)

# Metrics
rmse_train_lr = np.sqrt(mean_squared_error(y_train, y_train_pred_lr))
rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))

r2_train_lr = r2_score(y_train, y_train_pred_lr)
r2_test_lr = r2_score(y_test, y_test_pred_lr)


# =========================================
# 7. Polynomial Regression (Degree = 2)
# =========================================
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Predictions
y_train_pred_poly = poly_reg.predict(X_train_poly)
y_test_pred_poly = poly_reg.predict(X_test_poly)

# Metrics
rmse_train_poly = np.sqrt(mean_squared_error(y_train, y_train_pred_poly))
rmse_test_poly = np.sqrt(mean_squared_error(y_test, y_test_pred_poly))

r2_train_poly = r2_score(y_train, y_train_pred_poly)
r2_test_poly = r2_score(y_test, y_test_pred_poly)


# =========================================
# 8. Results Comparison Table
# =========================================
results = pd.DataFrame({
    "Model": [
        "Linear Regression (Degree 1)",
        "Polynomial Regression (Degree 2)"
    ],
    "Train RMSE": [rmse_train_lr, rmse_train_poly],
    "Test RMSE": [rmse_test_lr, rmse_test_poly],
    "Train R2": [r2_train_lr, r2_train_poly],
    "Test R2": [r2_test_lr, r2_test_poly]
})

print("\nModel Comparison Results:")
print(results)


# =========================================
# 9. Visualization - RMSE Comparison (Bar Plot)
# =========================================
rmse_df = pd.DataFrame({
    "Model": ["Degree 1", "Degree 1", "Degree 2", "Degree 2"],
    "Dataset": ["Train", "Test", "Train", "Test"],
    "RMSE": [
        rmse_train_lr,
        rmse_test_lr,
        rmse_train_poly,
        rmse_test_poly
    ]
})

plt.figure(figsize=(8, 6))
sns.barplot(data=rmse_df, x="Model", y="RMSE", hue="Dataset")
plt.title("RMSE Comparison Between Linear and Polynomial Models")
plt.show()


# =========================================
# 10. Actual vs Predicted Plot (Best Model: Degree 2)
# =========================================
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred_poly, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted (Polynomial Regression - Degree 2)")
plt.show()


# =========================================
# 11. Model Complexity vs Error (Overfitting Check)
# =========================================
degrees = range(1, 6)
train_errors = []
test_errors = []

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    
    train_errors.append(
        np.sqrt(mean_squared_error(y_train, train_pred))
    )
    test_errors.append(
        np.sqrt(mean_squared_error(y_test, test_pred))
    )

plt.figure(figsize=(8, 6))
plt.plot(degrees, train_errors, marker='o', label="Train RMSE")
plt.plot(degrees, test_errors, marker='o', label="Test RMSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("Model Complexity vs Error")
plt.legend()
plt.show()
