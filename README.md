# CodeAlpha Internship Tasks

This repository contains the machine learning projects completed as part of the CodeAlpha Internship Program.

The projects demonstrate practical applications of machine learning, data preprocessing, model training, evaluation, visualization, and prediction.

## Projects

### Task 1 – Credit Scoring Model

A machine learning project for predicting credit risk using the German Credit dataset.

**Technologies Used:**
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

**Models Used:**
- Logistic Regression
- Random Forest

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Project Folder:**
`CodeAlpha_CreditScoringModel`

---

### Task 2 – Disease Prediction

A machine learning classification project using medical diagnostic features to predict whether a tumor is malignant or benign.

**Technologies Used:**
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

**Models Used:**
- Logistic Regression
- Random Forest

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Project Folder:**
`CodeAlpha_DiseasePrediction`

> This project is intended for educational and machine learning demonstration purposes and is not a medical diagnostic tool.

---

### Task 3 – Handwritten Character Recognition

A deep learning project that uses a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset.

**Technologies Used:**
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

**Model:**
- Convolutional Neural Network (CNN)

**Features:**
- MNIST dataset loading
- Image normalization
- Image reshaping
- CNN model training
- Model evaluation
- Prediction generation
- Accuracy and loss visualization
- Trained model saving

**Project Folder:**
`CodeAlpha_HandwrittenCharacterRecognition`

---

## Repository Structure

```text
codealpha_tasks/
│
├── CodeAlpha_CreditScoringModel/
│   ├── data/
│   ├── results/
│   ├── credit_scoring.py
│   ├── requirements.txt
│   └── README.md
│
├── CodeAlpha_DiseasePrediction/
│   ├── results/
│   ├── disease_prediction.py
│   ├── requirements.txt
│   └── README.md
│
├── CodeAlpha_HandwrittenCharacterRecognition/
│   ├── results/
│   ├── handwritten_recognition.py
│   ├── handwritten_digit_cnn.keras
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
└── README.md