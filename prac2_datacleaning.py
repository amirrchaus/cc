import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
df = pd.read_csv("sample_data_50_100.csv")

# 4. Handling Data in Wrong Format
wrong_format_before = df.dtypes  # Store initial data types
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip()  # Removing leading/trailing spaces

wrong_format_after = df.dtypes  # Store data types after fixing
print("\nData Types Before Fixing Format:\n", wrong_format_before)
print("\nData Types After Fixing Format:\n", wrong_format_after)

# 5. Applying Scaling (Min-Max & Normalization)
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Store original data before scaling
original_data = df[numeric_cols].head()

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[numeric_cols] = scaler_minmax.fit_transform(df[numeric_cols])

# Standard Normalization
scaler_standard = StandardScaler()
df_standard = df.copy()
df_standard[numeric_cols] = scaler_standard.fit_transform(df[numeric_cols])

# Display transformation results
print("\nOriginal Data (Before Scaling):\n", original_data)
print("\nData after Min-Max Scaling:\n", df_minmax[numeric_cols].head())
print("\nData after Standard Normalization:\n", df_standard[numeric_cols].head())
