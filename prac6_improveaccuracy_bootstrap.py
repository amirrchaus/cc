#Boosting
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('dataDecision.csv')

# Preprocessing: Convert categorical variables to dummy variables and map 'YES'/'NO' to 1/0 df = pd.get_dummies(df, columns=['GENDER', 'AGE'], drop_first=True)
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Split the data into features (X) and target (y) X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting model
model = GradientBoostingClassifier(random_state=42) model.fit(X_train, y_train)

# Predictions and probabilities y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred) conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1) sensitivity = recall_score(y_test, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
standard_error = (accuracy * (1 - accuracy)) / len(y_test)

# Print results print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix) print("Precision:", precision)

print("Sensitivity (Recall):", sensitivity) print("Specificity:", specificity) print("Standard Error:", standard_error)

# Sample Input for prediction sample_input = {
'SMOKING': 2,
'YELLOW_FINGERS': 1,
'ANXIETY': 1,
'PEER_PRESSURE': 1,
'CHRONIC DISEASE': 1,
'FATIGUE': 2,
'ALLERGY': 1,
'WHEEZING': 1,
'ALCOHOL CONSUMING': 2,
'COUGHING': 2,
'SHORTNESS OF BREATH': 2, 'SWALLOWING DIFFICULTY': 1,
'CHEST PAIN': 2,
'GENDER_M': 1,
'AGE_Senior': 1,
'AGE_Youth': 0
}

# Convert sample input into DataFrame sample_df = pd.DataFrame([sample_input])

# Ensure all columns are present in the sample_df for col in X_train.columns:
if col not in sample_df.columns: sample_df[col] = 0

# Reorder the columns to match the training set sample_df = sample_df[X_train.columns]

# Make prediction
sample_prediction = model.predict(sample_df) sample_prediction_label = 'YES' if sample_prediction[0] == 1 else 'NO' print("\nSample Input Prediction:", sample_prediction_label)

# Plot Feature Importance (Boosting models can provide feature importance) plt.figure(figsize=(10, 6))
plt.barh(X.columns, model.feature_importances_) plt.xlabel('Feature Importance')

plt.title('Feature Importance from Gradient Boosting') plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob) # False Positive Rate, True Positive Rate roc_auc = auc(fpr, tpr) # Compute AUC

# Plot ROC Curve plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') # Diagonal line representing random model plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05]) plt.xlabel('False Positive Rate') plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve') plt.legend(loc="lower right")
plt.show()




#Bootstrap
import pandas as pd
from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r'C:\Users\student\Downloads\dataDecision.csv')

# One-hot encode categorical variables and map 'LUNG_CANCER' to binary df = pd.get_dummies(df, columns=['GENDER', 'AGE'], drop_first=True) df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Split dataset into features (X) and target (y) X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model (Bootstrap technique)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True) rf_model.fit(X_train, y_train)

# Make predictions and calculate probabilities y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics for Random Forest model accuracy_rf = accuracy_score(y_test, y_pred_rf) conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, zero_division=1) sensitivity_rf = recall_score(y_test, y_pred_rf)
specificity_rf = conf_matrix_rf[0, 0] / (conf_matrix_rf[0, 0] + conf_matrix_rf[0, 1]) if (conf_matrix_rf[0, 0] + conf_matrix_rf[0, 1]) > 0 else 0
standard_error_rf = (accuracy_rf * (1 - accuracy_rf)) / len(y_test)

# Print results for Random Forest model print("Random Forest Model Evaluation:") print(f"Accuracy: {accuracy_rf}") print(f"Confusion Matrix:\n{conf_matrix_rf}") print(f"Precision: {precision_rf}") print(f"Sensitivity (Recall): {sensitivity_rf}") print(f"Specificity: {specificity_rf}") print(f"Standard Error: {standard_error_rf}")

# Sample input for prediction sample_input = {
'SMOKING': 2,

'YELLOW_FINGERS': 1,
'ANXIETY': 1,
'PEER_PRESSURE': 1,
'CHRONIC DISEASE': 1,
'FATIGUE': 2,
'ALLERGY': 1,
'WHEEZING': 1,
'ALCOHOL CONSUMING': 2,
'COUGHING': 2,
'SHORTNESS OF BREATH': 2, 'SWALLOWING DIFFICULTY': 1,
'CHEST PAIN': 2,
'GENDER_M': 1,
'AGE_Senior': 1,
'AGE_Youth': 0
}

# Create DataFrame for sample input sample_df = pd.DataFrame([sample_input])

# Add missing columns from training data if not in sample input for col in X_train.columns:
if col not in sample_df.columns: sample_df[col] = 0

# Reorder columns to match training data sample_df = sample_df[X_train.columns]

# Make prediction for the sample input sample_prediction_rf = rf_model.predict(sample_df)
sample_prediction_label_rf = 'YES' if sample_prediction_rf[0] == 1 else 'NO' print("\nSample Input Prediction (Random Forest):", sample_prediction_label_rf)

# Plot feature importances for Random Forest plt.figure(figsize=(10, 6))
plt.barh(X.columns, rf_model.feature_importances_) plt.xlabel('Feature Importance')
plt.title('Feature Importances in Random Forest') plt.show()

# Calculate ROC curve and AUC for Random Forest model
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_prob_rf) # False Positive Rate, True Positive Rate
roc_auc_rf = auc(fpr_rf, tpr_rf) # Compute AUC

# Plot ROC Curve for Random Forest model plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})') plt.plot([0, 1], [0, 1], color='gray', linestyle='--') # Diagonal line representing random model plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05]) plt.xlabel('False Positive Rate') plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest') plt.legend(loc="lower right")
plt.show()



#ALL
import pandas as pd import numpy as np import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier from sklearn.utils import resample
from scipy.stats import chi2_contingency

# Load data
df = pd.read_csv(r'C:\Users\student\Downloads\dataDecision.csv')

# Data Cleaning
df = df.dropna() # Drop missing values
df = pd.get_dummies(df, columns=['GENDER', 'AGE'], drop_first=True) df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Exploratory Data Analysis print("Data Head:") print(df.head())

# Correlation Heatmap plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f") plt.title("Correlation Heatmap")
plt.show()

# # Chi-Square Test for Categorical Features
# cat_cols = ['GENDER', 'AGE',] # Replace with actual column names # for col in cat_cols:
#	contingency_table = pd.crosstab(df[col], df['LUNG_CANCER']) #	chi2, p, dof, expected = chi2_contingency(contingency_table) #	print(f"Chi-Square Test for {col}:")
#	print(f" Chi2: {chi2:.3f}, p-value: {p:.3f}")

# Feature Selection and Target Variable X = df.drop(columns=['LUNG_CANCER']) y = df['LUNG_CANCER']

# Scaling
standard_scaler = StandardScaler() minmax_scaler = MinMaxScaler()

X_std = standard_scaler.fit_transform(X) X_mm = minmax_scaler.fit_transform(X)

# Train/test split
X_train_std, X_test_std, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=69)
X_train_mm, X_test_mm, _, _ = train_test_split(X_mm, y, test_size=0.2, random_state=69)

# Simple PCA
pca = PCA(n_components=5)
X_train_std_pca = pca.fit_transform(X_train_std) X_test_std_pca = pca.transform(X_test_std) X_train_mm_pca = pca.fit_transform(X_train_mm) X_test_mm_pca = pca.transform(X_test_mm)

# Evaluation function
def evaluate_model(y_true, y_pred, y_prob): acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=1) sens = recall_score(y_true, y_pred)
spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) else 0 se = (acc * (1 - acc)) / len(y_true)
fpr, tpr, _ = roc_curve(y_true, y_prob) model_auc = auc(fpr, tpr)
return acc, cm, prec, sens, spec, se, model_auc

# Models
dt_bag = BaggingClassifier(DecisionTreeClassifier(random_state=69), n_estimators=50, random_state=69)
nb_bag = BaggingClassifier(GaussianNB(), n_estimators=50, random_state=69) lr = LinearRegression()

# Boosting model
dt_ada = AdaBoostClassifier(DecisionTreeClassifier(random_state=69), n_estimators=50, random_state=69)

# Bootstrap Sampling
X_bootstrap, y_bootstrap = resample(X, y, replace=True, random_state=69)

# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=69) def cross_val_accuracy(model, X_data, y_data):
scores = cross_val_score(model, X_data, y_data, cv=cv, scoring='accuracy') return np.mean(scores)

# Train and evaluate (Standardized)

dt_bag.fit(X_train_std, y_train) nb_bag.fit(X_train_std, y_train) lr.fit(X_train_std, y_train) dt_ada.fit(X_train_std, y_train)

pred_dt_bag = dt_bag.predict(X_test_std) pred_nb_bag = nb_bag.predict(X_test_std) pred_lr = lr.predict(X_test_std)
pred_lr_bin = (pred_lr > 0.5).astype(int) pred_dt_ada = dt_ada.predict(X_test_std)

res_std = {
'Bagging DT': evaluate_model(y_test, pred_dt_bag, dt_bag.predict_proba(X_test_std)[:, 1]), 'Bagging NB': evaluate_model(y_test, pred_nb_bag, nb_bag.predict_proba(X_test_std)[:, 1]), 'Linear Regression': evaluate_model(y_test, pred_lr_bin, pred_lr),
'AdaBoost DT': evaluate_model(y_test, pred_dt_ada, dt_ada.predict_proba(X_test_std)[:, 1])
}

# Cross-validation accuracy comparison (Bagging DT example) dt_cv_std = cross_val_accuracy(dt_bag, X_std, y)

# Print results
print("Model Evaluation Results (Standardized)\n") for model_name in res_std:
acc_std, cm_std, prec_std, sens_std, spec_std, se_std, auc_std = res_std[model_name] print(f"{model_name}:")
print(f" Accuracy: {acc_std:.3f}") print(f" Confusion Matrix:\n{cm_std}") print(f" Precision: {prec_std:.3f}") print(f" Sensitivity: {sens_std:.3f}") print(f" Specificity: {spec_std:.3f}") print(f" Standard Error: {se_std:.4f}") print(f" AUC: {auc_std:.3f}\n")

print("Cross-Validation Accuracy (Bagging DT example):") print(f" Standard Scaler: {dt_cv_std:.3f}")





