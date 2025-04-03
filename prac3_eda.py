import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("car_sales_data.csv")

# Convert categorical variables to numerical using encoding
df_encoded = df.copy()
df_encoded = pd.get_dummies(df_encoded, columns=['Car_Brand'], drop_first=True)

# Basic Info
print("Dataset Overview:\n", df.info())
print("\nSummary Statistics:\n", df.describe())

# Handling Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)

# Removing Duplicates
df.drop_duplicates(inplace=True)

# Mean & Median
print("\nMean Values:\n", df.mean(numeric_only=True))
print("\nMedian Values:\n", df.median(numeric_only=True))

# Heatmap (Only for numerical columns)
plt.figure(figsize=(10,6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap of Feature Correlations")
plt.show()

# Pie Chart
plt.figure(figsize=(7,7))
df['Car_Brand'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title("Car Brand Distribution")
plt.ylabel("")
plt.show()

# Bar Chart
plt.figure(figsize=(10,5))
sns.barplot(x=df['Car_Brand'], y=df['Sales'], estimator=np.mean)
plt.title("Average Sales by Car Brand")
plt.xticks(rotation=45)
plt.show()

# Line Chart
plt.figure(figsize=(10,5))
sns.lineplot(x=df['Year'], y=df['Sales'], estimator=np.mean, errorbar=None)
plt.title("Yearly Sales Trend")
plt.show()

# Doughnut Chart
plt.figure(figsize=(7,7))
colors = sns.color_palette('pastel')
wedges, texts, autotexts = plt.pie(df['Car_Brand'].value_counts(), autopct='%1.1f%%', colors=colors, pctdistance=0.85)
plt.gca().add_artist(plt.Circle((0,0), 0.7, fc='white'))
plt.title("Car Brand Distribution (Doughnut Chart)")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Price'], y=df['Sales'], hue=df['Car_Brand'])
plt.title("Sales vs. Price")
plt.show()

# Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Car_Brand'], y=df['Sales'])
plt.title("Sales Distribution by Brand")
plt.xticks(rotation=45)
plt.show()

# Pair Plot
sns.pairplot(df_encoded)
plt.suptitle("Pair Plot of Features", y=1.02)
plt.show()

# Violin Chart
plt.figure(figsize=(8,5))
sns.violinplot(x=df['Car_Brand'], y=df['Sales'])
plt.title("Sales Distribution by Brand")
plt.xticks(rotation=45)
plt.show()

# Joint Plot
sns.jointplot(x=df['Price'], y=df['Sales'], kind='hex')
plt.suptitle("Joint Plot of Price vs Sales", y=1.02)
plt.show()

# Scatter with Trend Line
plt.figure(figsize=(8,5))
sns.regplot(x=df['Advertising_Budget'], y=df['Sales'])
plt.title("Sales vs. Advertising Budget (With Trend Line)")
plt.show()

# Histogram of Defect Rate
plt.figure(figsize=(8,5))
sns.histplot(df['Defect_Rate'], bins=10, kde=True)
plt.title("Histogram of Defect Rate")
plt.show()
