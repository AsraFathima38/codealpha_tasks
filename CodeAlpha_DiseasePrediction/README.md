# Disease Prediction Using Machine Learning

## CodeAlpha Machine Learning Internship – Task 4

This project uses machine learning classification techniques to predict whether a breast tumor is classified as **malignant** or **benign** based on numerical tumor measurements.

The project compares Logistic Regression and Random Forest models and evaluates their performance using multiple classification metrics.

## Objective

The main objectives of this project are:

- Load and explore a medical dataset.
- Preprocess and scale the data.
- Train machine learning classification models.
- Predict tumor classification.
- Evaluate model performance.
- Compare different models.
- Identify important features influencing Random Forest predictions.

## Dataset

The project uses the **Breast Cancer Wisconsin dataset** available through scikit-learn.

The dataset contains:

- 569 samples
- 30 numerical features
- 2 target classes:
  - Malignant
  - Benign

The features describe measurements related to cell nuclei, such as radius, texture, perimeter, area, concavity and symmetry.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Machine Learning Models

Two classification algorithms were used:

### 1. Logistic Regression

Used as a classification model after standardizing the numerical features.

### 2. Random Forest

Used as an ensemble classification model and also to analyze feature importance.

## Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
DataFrame Creation
   ↓
Feature and Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Visualization
   ↓
Feature Importance