import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------- STEP 1: Load and Preprocess Data -----------------------------

# Load dataset from CSV (Ensure "car_sales_data.csv" is in the working directory)
data = pd.read_csv("car_sales_data.csv")

# Drop categorical columns (if any) and keep only numeric data
data = data.select_dtypes(include=['number'])

# Compute correlation coefficient
correlation_matrix = data.corr()
correlation_coefficient = correlation_matrix.loc["Advertisement_Spend", "Sales"]

print("\nCorrelation Coefficient between Ad Spend and Sales:", correlation_coefficient)

# Extract feature (independent variable) and target (dependent variable)
X = data[['Advertisement_Spend']]  # Rating as independent variable
y = data['Sales']  # Car Price as dependent variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------- STEP 2: Train Linear Regression Model -----------------------------

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# ----------------------------- STEP 3: Compute Error & Goodness of Fit -----------------------------

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Linear Regression Model Coefficient (a): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")
print(f"Linear Equation: y = {model.coef_[0]}x + {model.intercept_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared (Goodness of Fit): {r2}")

# ----------------------------- STEP 4: Visualization -----------------------------

# Plot actual data and regression line
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color="blue", label="Actual Data")
plt.plot(X_test, y_pred, color="red", label="Regression Line")

# Generate and plot a shaded region for prediction confidence
X_range = np.linspace(X_test.min().values[0], X_test.max().values[0], 100).reshape(-1, 1)
y_range = model.predict(X_range)
plt.fill_between(
    X_range.flatten(),
    y1=(y_range - 0.1 * y_range),
    y2=(y_range + 0.1 * y_range),
    color="gray",
    alpha=0.3,
    label="Shaded Region"
)

# Label the plot
plt.xlabel("Advertisement_Spend")
plt.ylabel("Sales")
plt.title("Advertisement_Spend vs Sales")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------- STEP 5: User Input Prediction -----------------------------

# Predict for a user-provided value
X_input = float(input(f"\nEnter a Advertisement_Spend value (e.g., 4.5): "))
y_pred_input = model.predict(pd.DataFrame({"Advertisement_Spend": [X_input]}))
print(f"The predicted Car Price for Advertisement_Spend {X_input} is {y_pred_input[0]}")
