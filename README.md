# 💳 Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Fraud_Detection-orange)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-purple)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI/CD-success)

An end-to-end Machine Learning system for detecting fraudulent credit card transactions using **XGBoost**, **SMOTE**, **FastAPI**, **MLflow**, **Docker**, and **SHAP Explainability**.

---

## 🚀 Features

* Fraud Detection using XGBoost
* Class Imbalance Handling using SMOTE
* REST API built with FastAPI
* Explainable AI using SHAP
* Precision-Recall Optimization
* MLflow Experiment Tracking
* Dockerized Deployment
* GitHub Actions CI/CD Pipeline
* Production-Ready Project Structure

---

## 📌 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent less than **0.2%** of all transactions.

The goal is to build a scalable machine learning pipeline capable of detecting fraudulent transactions while minimizing false negatives and maintaining strong precision.

---

## 🏗 System Architecture

```text
Credit Card Dataset
        │
        ▼
 Data Preprocessing
        │
        ▼
 Train/Test Split
        │
        ▼
      SMOTE
        │
        ▼
  XGBoost Model
        │
        ▼
 Model Evaluation
        │
        ├── Confusion Matrix
        ├── Precision-Recall Curve
        └── SHAP Analysis
        │
        ▼
 MLflow Tracking
        │
        ▼
 FastAPI Service
        │
        ▼
 REST Prediction API
```

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
│
├── models/
│   └── fraud_model.pkl
│
├── assets/
│   ├── confusion_matrix.png
│   ├── pr_curve.png
│   └── shap_summary.png
│
├── mlruns/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.6591 |
| Recall    | 0.8878 |
| F1 Score  | 0.7565 |
| ROC-AUC   | 0.9828 |
| PR-AUC    | 0.8755 |

The model prioritizes high recall to maximize fraud detection while maintaining strong overall performance.

---

## 📉 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

---

## 📈 Precision-Recall Curve

![PR Curve](assets/pr_curve.png)

---

## 🔍 Explainable AI (SHAP)

SHAP helps explain model predictions and identify the most influential features.

### Top Influential Features

* V14
* V4
* V17
* V10
* V3
* V12

### SHAP Summary Plot

![SHAP Summary](assets/shap_summary.png)

---

## 📊 MLflow Experiment Tracking

The project uses MLflow to track:

* Hyperparameters
* Training Metrics
* Model Artifacts
* Experiment History

### Start MLflow UI

```bash
mlflow ui --workers 1
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🚀 FastAPI Deployment

### Start API

```bash
uvicorn api.app:app --reload
```

### Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

| Method | Endpoint | Description      |
| ------ | -------- | ---------------- |
| GET    | /        | API Status       |
| GET    | /health  | Health Check     |
| POST   | /predict | Fraud Prediction |

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Subhajitdas99/fraud_detection_System.git
cd fraud_detection_System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🐳 Docker

### Build Image

```bash
docker build -t fraud-detection .
```

### Run Container

```bash
docker run -p 8000:8000 fraud-detection
```

---

## 🧪 Run Training

```bash
python src/fraud_detection.py
```

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)
* SHAP
* FastAPI
* Uvicorn
* MLflow
* Docker
* GitHub Actions

---

## 🔮 Future Improvements

* Model Registry
* Real-Time Fraud Monitoring
* Kafka Streaming Integration
* Cloud Deployment (AWS / Azure)
* Monitoring Dashboard
* Automated Retraining Pipeline

---

## 👨‍💻 Author

### Subhajit Das

B.Tech CSE (AI & ML)

Machine Learning Engineer | Data Science | AI Engineering

GitHub: https://github.com/Subhajitdas99




