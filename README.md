# 💳 Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Fraud_Detection-orange)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI/CD-success)

An end-to-end Machine Learning system for detecting fraudulent credit card transactions using **XGBoost**, **SMOTE**, **FastAPI**, **Docker**, and **SHAP Explainability**.

---

## 🚀 Key Features

* Fraud Detection using XGBoost
* Class Imbalance Handling with SMOTE
* REST API built with FastAPI
* Explainable AI using SHAP
* Precision-Recall Optimization
* Dockerized Deployment
* GitHub Actions CI/CD
* Production-Ready Project Structure

---

## 📌 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions account for less than 0.2% of all transactions.

The objective of this project is to build a scalable machine learning pipeline capable of identifying fraudulent transactions while minimizing false negatives.

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
        ├── PR Curve
        └── SHAP Analysis
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
├── src/
│   ├── fraud_detection.py
│   ├── evaluate.py
│   └── shap_analysis.py
├── tests/
├── models/
│   └── fraud_model.pkl
├── data/
│   └── creditcard.csv
├── assets/
│   ├── confusion_matrix.png
│   ├── pr_curve.png
│   └── shap_summary.png
├── Dockerfile
├── requirements.txt
└── README.md
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

The model achieves high recall, ensuring that most fraudulent transactions are successfully detected.

---

## 📉 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

---

## 📈 Precision-Recall Curve

![PR Curve](assets/pr_curve.png)

---

## 🔍 SHAP Explainability

Top Influential Features:

* V14
* V4
* V17
* V10
* V3
* V12

### SHAP Summary Plot

![SHAP Summary](assets/shap_summary.png)

---

## 🚀 API Usage

Start the API:

```bash
uvicorn api.app:app --reload
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
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
* GitHub Actions

---

## 🔮 Future Improvements

* MLflow Experiment Tracking
* Model Registry
* Kafka Integration
* Real-Time Fraud Monitoring
* Cloud Deployment
* Monitoring Dashboard

---

## 👨‍💻 Author

**Subhajit Das**

B.Tech CSE (AI & ML)

Machine Learning Engineer | Data Science | AI Engineering



