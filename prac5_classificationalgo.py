import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


#NaiveBayes
df = pd.read_csv("C:\\Users\\student\\Downloads\\diabetes.csv")

print(df.head())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
print(y_pred)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Confusion matrix visualization
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")  # Random classifier (diagonal line)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()









#PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "C:/Users/student/Downloads/myData.csv"
data = pd.read_csv(file_path)

# Ensure the target variable is binary
# If Outcome has continuous values, convert them to binary (e.g., threshold at 0.5)
data['Outcome1'] = data['Outcome'].apply(lambda x: 1 if x >= 0.5 else 0)

# Define features and target variable
X = data.drop(columns=['Outcome'])  # Features
y = data['Outcome1']  # Target

# Scale the features before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (you can adjust the number of components or use 95% variance explained)
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)

# Specificity calculation
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
specificity = TN / (TN + FP)

# Standard error
standard_error = np.sqrt((accuracy * (1 - accuracy)) / len(y_test))

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

# Plot the Decision Tree (limit to 5 levels)
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=[f'PC{i+1}' for i in range(X_pca.shape[1])], 
          class_names=['0', '1'], filled=True, max_depth=3)
plt.title("Decision Tree (Max Depth = 5)")
plt.show()

# Print results
print("Predictions:", y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Standard Error:", standard_error)
print("Precision:", precision)
print("Sensitivity (Recall):", sensitivity)
print("Specificity:", specificity)
print("AUC:", roc_auc)






##Linear Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Linear Regression-like Classification
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "C:\\Users\\student\\Downloads\\diabetes.csv"
data = pd.read_csv(file_path)

# Ensure the target variable is binary
# If Outcome has continuous values, convert them to binary (e.g., threshold at 0.5)
data['Outcome1'] = data['Outcome'].apply(lambda x: 1 if x >= 0.5 else 0)

# Define features and target variable
X = data.drop(columns=['Outcome'])  # Features
y = data['Outcome1']  # Target

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression (Logistic Regression)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)  # Train the model

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_prob_lr = lr.predict_proba(X_test)[:, 1]

# Metrics for Linear Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
sensitivity_lr = recall_score(y_test, y_pred_lr)

# Specificity for Linear Regression
TN_lr = conf_matrix_lr[0, 0]
FP_lr = conf_matrix_lr[0, 1]
specificity_lr = TN_lr / (TN_lr + FP_lr)

# Standard error for Linear Regression
standard_error_lr = np.sqrt((accuracy_lr * (1 - accuracy_lr)) / len(y_test))

# ROC curve and AUC for Linear Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Plot ROC Curve for Linear Regression
plt.figure()
plt.plot(fpr_lr, tpr_lr, color="green", lw=2, label=f"Linear Regression (AUC = {roc_auc_lr:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Linear Regression")
plt.legend(loc="lower right")
plt.show()

# Print Metrics
print("===== Linear Regression Metrics =====")
print("Accuracy:", accuracy_lr)
print("Confusion Matrix:\n", conf_matrix_lr)
print("Standard Error:", standard_error_lr)
print("Precision:", precision_lr)
print("Sensitivity (Recall):", sensitivity_lr)
print("Specificity:", specificity_lr)
print("AUC:", roc_auc_lr)
