import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
stratify=y,
random_state=42,
)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.savefig(
OUTPUT_DIR / "confusion_matrix.png",
bbox_inches="tight"
)

plt.close()

print("Saved confusion_matrix.png")


