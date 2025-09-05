import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("creditcard.csv")

# Features & target
X = df.drop(columns=['Class'])
y = df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------
# 2. Handle imbalance
# -------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -------------------------
# 3. Build pipeline with XGBoost
# -------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=1,  # SMOTE already balances classes
        random_state=42
    ))
])

# -------------------------
# 4. Train model
# -------------------------
pipeline.fit(X_train_res, y_train_res)

# -------------------------
# 5. Evaluate model
# -------------------------
y_probs = pipeline.predict_proba(X_test)[:, 1]

# Threshold tuning (default 0.5, can adjust later)
threshold = 0.5
y_pred = (y_probs >= threshold).astype(int)

print("Classification Report (threshold = 0.5):")
print(classification_report(y_test, y_pred))

# Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.4f}")

# Plot PR curve
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# -------------------------
# 6. Save trained model
# -------------------------
joblib.dump(pipeline, "fraud_model.pkl")
print("✅ Full-feature XGBoost model trained and saved as fraud_model.pkl")

