import pandas as pd

# Load the German Credit dataset
data = pd.read_csv("data/german.data", sep=" ", header=None)

# Give meaningful names to the columns
columns = [
    "checking_account",
    "duration",
    "credit_history",
    "loan_purpose",
    "credit_amount",
    "savings_account",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "credit_risk"
]

data.columns = columns

# Separate features and target
X = data.drop("credit_risk", axis=1)
y = data["credit_risk"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True)

# Display the processed data
print("\nProcessed data:")
print(X.head())

print("\nProcessed data shape:", X.shape)


# --------------------------------------------------
# Split the dataset into training and testing data
# --------------------------------------------------

from sklearn.model_selection import train_test_split

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


# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

from sklearn.preprocessing import StandardScaler

# Create the scaler
scaler = StandardScaler()

# Fit scaler only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaling to testing data
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed successfully!")


# --------------------------------------------------
# Logistic Regression Model
# --------------------------------------------------

from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model using scaled training data
model.fit(X_train_scaled, y_train)

print("\nScaled Logistic Regression model trained successfully!")


# --------------------------------------------------
# Make Predictions
# --------------------------------------------------

# Make predictions using scaled testing data
y_pred = model.predict(X_test_scaled)

print("\nPredictions made successfully!")
print("First 10 predictions:", y_pred[:10])


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=2
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=2
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=2
)

# Get prediction probabilities for ROC-AUC
y_prob = model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)


# --------------------------------------------------
# Display Results
# --------------------------------------------------

print("\n--- Logistic Regression Model Evaluation ---")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))
# --------------------------------------------------
# Random Forest Model
# --------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# Make predictions
rf_pred = rf_model.predict(X_test)

# Get prediction probabilities
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
rf_accuracy = accuracy_score(y_test, rf_pred)

rf_precision = precision_score(
    y_test,
    rf_pred,
    pos_label=2
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    pos_label=2
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    pos_label=2
)

rf_roc_auc = roc_auc_score(
    y_test,
    rf_prob
)

# Display results
print("\n--- Random Forest Model Evaluation ---")
print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1-Score :", round(rf_f1, 4))
print("ROC-AUC  :", round(rf_roc_auc, 4))
# --------------------------------------------------
# Confusion Matrix for Final Model
# --------------------------------------------------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Good Credit", "Bad Credit"]
)

disp.plot()

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

# Save the figure
plt.savefig("results/confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved successfully!")
# --------------------------------------------------
# ROC Curve for Logistic Regression
# --------------------------------------------------

from sklearn.metrics import roc_curve

# Calculate ROC curve values
# --------------------------------------------------
# ROC Curve for Logistic Regression
# --------------------------------------------------


# Calculate ROC curve values
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob,
    pos_label=2
)

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
# --------------------------------------------------
# Random Forest Feature Importance
# --------------------------------------------------

# Get feature importance from Random Forest
feature_importance = rf_model.feature_importances_

# Create a DataFrame
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

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

# Save the graph
plt.savefig("results/feature_importance.png")

plt.show()

print("\nFeature importance graph saved successfully!")
# --------------------------------------------------
# Save Model Results
# --------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        rf_accuracy
    ],
    "Precision": [
        precision,
        rf_precision
    ],
    "Recall": [
        recall,
        rf_recall
    ],
    "F1-Score": [
        f1,
        rf_f1
    ],
    "ROC-AUC": [
        roc_auc,
        rf_roc_auc
    ]
})

# Save results to CSV
results.to_csv(
    "results/model_results.csv",
    index=False
)

print("\nModel results saved successfully!")
print(results)