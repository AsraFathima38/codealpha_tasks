# Credit Scoring Model

## CodeAlpha Machine Learning Internship - Task 1

A machine learning project that predicts the creditworthiness of applicants using financial and personal information from the German Credit dataset.

## Project Overview

Credit scoring helps financial institutions evaluate whether a customer is likely to be a good or bad credit risk.

In this project, machine learning classification algorithms are used to predict credit risk based on factors such as credit amount, loan duration, age, checking account status, savings information, and credit history.

## Dataset

The project uses the **Statlog German Credit Data** dataset from the UCI Machine Learning Repository.

- Total records: 1000
- Input features: 20
- Target: Credit Risk
- Good Credit: 1
- Bad Credit: 2

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Machine Learning Models

Two classification algorithms were implemented:

1. Logistic Regression
2. Random Forest Classifier

Feature scaling was applied before training the Logistic Regression model.

## Data Preprocessing

The following preprocessing steps were performed:

- Loaded the German Credit dataset
- Assigned meaningful column names
- Separated features and target variable
- Converted categorical features into numerical features using one-hot encoding
- Split the dataset into training and testing sets
- Applied feature scaling for Logistic Regression

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.785 | 0.6604 | 0.5833 | 0.6195 | 0.8000 |
| Random Forest | 0.770 | 0.7188 | 0.3833 | 0.5000 | 0.7969 |

Logistic Regression achieved the highest overall accuracy, F1-score, and ROC-AUC among the two models.

## Visualizations

The project generates the following visualizations:

### Confusion Matrix

Shows the correct and incorrect predictions made by the Logistic Regression model.

### ROC Curve

Shows the classification performance of the Logistic Regression model.

### Model Comparison

Compares the performance of Logistic Regression and Random Forest using different evaluation metrics.

### Feature Importance

Shows the top features that contributed to the Random Forest predictions.

## Important Features

The top features identified by the Random Forest model include:

- Credit Amount
- Age
- Loan Duration
- Checking Account Status
- Installment Rate
- Residence Since
- Other Installment Plans
- Credit History
- Savings Account

## Project Structure

```text
CodeAlpha_CreditScoringModel
│
├── data
│   └── german.data
│
├── results
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── model_results.csv
│   └── roc_curve.png
│
├── credit_scoring.py
├── requirements.txt
└── README.md