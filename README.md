# 💳 Fraud Detection System

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using XGBoost, SMOTE, FastAPI, and Explainable AI (SHAP).

---

## 📌 Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent less than 0.2% of all transactions.

This project builds a production-ready fraud detection pipeline that:

* Detects fraudulent transactions using XGBoost
* Handles class imbalance with SMOTE
* Provides REST API inference using FastAPI
* Explains model predictions using SHAP
* Generates evaluation visualizations
* Supports Docker deployment

---

## 🏗 Project Architecture

Data Collection
↓
Preprocessing
↓
Train/Test Split
↓
SMOTE Balancing
↓
XGBoost Training
↓
Model Evaluation
↓
SHAP Explainability
↓
FastAPI Deployment

---

## 📂 Project Structure

```text
fraud_detection_System/
│
├── api/
│   └── app.py
│
├── src/
│   ├── fraud_detection.py
│   ├── evaluate.py
│   └── shap_analysis.py
│
├── tests/
│   ├── test_single_api.py
│   ├── batch_test_api_full.py
│   └── generate_test_csv.py
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── fraud_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── pr_curve.png
│   └── shap_summary.png
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

Dataset: Credit Card Fraud Detection Dataset

Features:

* Time
* V1–V28 (PCA transformed features)
* Amount
* Class (Target)

Class Distribution:

* Non-Fraud: 284,315
* Fraud: 492

This severe imbalance makes fraud detection challenging.

---

## 🤖 Model

Algorithm:

* XGBoost Classifier

Class Imbalance Handling:

* SMOTE (Synthetic Minority Oversampling Technique)

Pipeline:

1. Data Loading
2. Train/Test Split
3. SMOTE Balancing
4. Standard Scaling
5. XGBoost Training
6. Threshold-Based Prediction

---

## 📈 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.6591 |
| Recall    | 0.8878 |
| F1 Score  | 0.7565 |
| ROC-AUC   | 0.9828 |
| PR-AUC    | 0.8755 |

These results demonstrate strong fraud detection performance while maintaining high recall for fraudulent transactions.

---

## 📉 Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

---

## 📈 Precision Recall Curve

![PR Curve](outputs/pr_curve.png)

---

## 🔍 Explainable AI (SHAP)

SHAP is used to explain feature importance and model behavior.

Top Influential Features:

* V14
* V4
* V17
* V10
* V3
* V12

SHAP Summary Plot:

![SHAP Summary](outputs/shap_summary.png)

---

## 🚀 FastAPI Deployment

Run API:

```bash
uvicorn api.app:app --reload
```

Open Swagger Docs:

```text
http://127.0.0.1:8000/docs
```

Available Endpoints:

### GET /

Returns API status.

### GET /health

Health check endpoint.

### POST /predict

Predict whether a transaction is fraudulent.

---

## Example Request

```json
{
  "Time": 0,
  "V1": 0,
  "V2": 0,
  "V3": 0,
  "V4": 0,
  "V5": 0,
  "V6": 0,
  "V7": 0,
  "V8": 0,
  "V9": 0,
  "V10": 0,
  "V11": 0,
  "V12": 0,
  "V13": 0,
  "V14": 0,
  "V15": 0,
  "V16": 0,
  "V17": 0,
  "V18": 0,
  "V19": 0,
  "V20": 0,
  "V21": 0,
  "V22": 0,
  "V23": 0,
  "V24": 0,
  "V25": 0,
  "V26": 0,
  "V27": 0,
  "V28": 0,
  "Amount": 100,
  "threshold": 0.5
}
```

---

## 🐳 Docker

Build:

```bash
docker build -t fraud-detection .
```

Run:

```bash
docker run -p 8000:8000 fraud-detection
```

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SMOTE
* SHAP
* FastAPI
* Uvicorn
* Docker
* GitHub

---

## Future Improvements

* MLflow Experiment Tracking
* Model Registry
* CI/CD Enhancements
* Real-Time Streaming Fraud Detection
* Kafka Integration
* Cloud Deployment
* Monitoring Dashboard

---

## Author

Subhajit Das

B.Tech CSE (AI & ML)

Machine Learning | Data Science | AI Engineering


