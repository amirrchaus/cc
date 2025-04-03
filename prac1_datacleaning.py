import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("sample_data_50_100.csv")

# Display initial dataset info
df.info()
print("\nInitial Dataset:\n", df.head())

# 1. Handling Missing Values
missing_values_before = df.isnull().sum()
df.fillna(df.median(numeric_only=True), inplace=True)  # Filling missing values with median
missing_values_after = df.isnull().sum()
print("\nMissing Values Before Cleaning:\n", missing_values_before)
print("\nMissing Values After Cleaning:\n", missing_values_after)

# 2. Removing Duplicates
duplicates_before = df[df.duplicated()]
print("\nDuplicate Rows Before Cleaning:\n", duplicates_before)
df.drop_duplicates(inplace=True)
duplicates_after = df.duplicated().sum()
print("\nDuplicate Rows After Cleaning:", duplicates_after)

# 3. Identifying and Removing Noisy Data
# Binning method for noise reduction
num_bins = 5
for col in df.select_dtypes(include=[np.number]).columns:
    df[f'{col}_Binned'] = pd.cut(df[col], bins=num_bins, labels=False)

# Boxplot for detecting outliers for all numerical columns
outliers = {}
for col in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(8, 5))
    plt.boxplot(df[col], vert=False, patch_artist=True)
    plt.title(f"Boxplot of {col} to Identify Outliers")
    plt.xlabel(col)
    plt.show()
    
    # Identifying outliers using IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

print("\nOutliers Identified:\n", outliers)

# Removing outliers using IQR for all numerical columns
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Display cleaned dataset info
df.info()
print("\nCleaned Dataset:\n", df.head())
