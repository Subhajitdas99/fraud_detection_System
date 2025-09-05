import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import Logisticregression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard.csv")

df = df.sample(n=20000, random_state=42)

pipeline = pipeline([
    ("scaler", StandardScaler()),
    ("model", Logisticregression(max_iter=500, class_weight="balanced"))
])