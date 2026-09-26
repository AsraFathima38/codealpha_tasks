from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the breast cancer dataset
data = load_breast_cancer()

# Convert the dataset into a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target column
df["target"] = data.target

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset shape:", df.shape)

# Display target classes
print("\nTarget classes:")
print(data.target_names)
# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget value counts:")
print(y.value_counts())
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed successfully!")
from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully!")
# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

print("\nPredictions made successfully!")
print("First 10 predictions:", y_pred[:10])
print("First 10 actual values:", y_test.values[:10])
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Get prediction probabilities for ROC-AUC
y_prob = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

# Display the results
print("\n--- Logistic Regression Model Evaluation ---")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))
from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# Make predictions
rf_pred = rf_model.predict(X_test)

# Get prediction probabilities
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_prob)

print("\n--- Random Forest Model Evaluation ---")
print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1-Score :", round(rf_f1, 4))
print("ROC-AUC  :", round(rf_roc_auc, 4))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Malignant", "Benign"]
)

disp.plot()

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

# Save the confusion matrix
plt.savefig("results/confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved successfully!")
from sklearn.metrics import roc_curve

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.2f})"
)

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression - ROC Curve")
plt.legend()

plt.tight_layout()

# Save ROC curve
plt.savefig("results/roc_curve.png")

plt.show()

print("\nROC curve saved successfully!")
# Model comparison
import numpy as np

models = ["Logistic Regression", "Random Forest"]

accuracy_scores = [accuracy, rf_accuracy]
precision_scores = [precision, rf_precision]
recall_scores = [recall, rf_recall]
f1_scores = [f1, rf_f1]
roc_auc_scores = [roc_auc, rf_roc_auc]

x = np.arange(len(models))
width = 0.15

plt.figure(figsize=(10, 6))

plt.bar(x - 2*width, accuracy_scores, width, label="Accuracy")
plt.bar(x - width, precision_scores, width, label="Precision")
plt.bar(x, recall_scores, width, label="Recall")
plt.bar(x + width, f1_scores, width, label="F1-Score")
plt.bar(x + 2*width, roc_auc_scores, width, label="ROC-AUC")

plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Disease Prediction Model Performance Comparison")

plt.xticks(x, models)
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()

# Save the comparison graph
plt.savefig("results/model_comparison.png")

plt.show()

print("\nModel comparison graph saved successfully!")
# Random Forest Feature Importance

feature_importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# Display top 10 important features
print("\n--- Top 10 Important Features ---")
print(importance_df.head(10))

# Plot top 10 features
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df.head(10)["Feature"][::-1],
    importance_df.head(10)["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features - Random Forest")

plt.tight_layout()

# Save feature importance graph
plt.savefig("results/feature_importance.png")

plt.show()

print("\nFeature importance graph saved successfully!")
# Save model evaluation results to CSV

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy, rf_accuracy],
    "Precision": [precision, rf_precision],
    "Recall": [recall, rf_recall],
    "F1-Score": [f1, rf_f1],
    "ROC-AUC": [roc_auc, rf_roc_auc]
})

results.to_csv("results/model_results.csv", index=False)

print("\nModel results saved successfully!")
print(results)
# Sample disease prediction

# Select one sample from the test dataset
sample = X_test.iloc[0]

# Scale the sample using the same scaler
sample_scaled = scaler.transform(sample.to_frame().T)

# Make prediction
sample_prediction = model.predict(sample_scaled)[0]

# Get model probability
sample_probability = model.predict_proba(sample_scaled)[0].max()

print("\n--- Sample Disease Prediction ---")

if sample_prediction == 0:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")

print("Model Probability:", round(sample_probability * 100, 2), "%")

print("Note: This is an educational ML demonstration, not a medical diagnosis.")