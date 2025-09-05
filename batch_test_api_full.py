import pandas as pd
import numpy as np
import requests

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Load CSV (must contain V1-V28, Amount, Time columns)
df = pd.read_csv("test_transactions_full.csv")

results = []

for idx, row in df.iterrows():
    payload = {col: row[col] for col in df.columns}  # include all features
    payload["threshold"] = 0.5  # can adjust threshold per request

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        res = response.json()
        results.append({**row.to_dict(), **res})
    else:
        print(f"⚠️ Error on row {idx}: {response.text}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("api_results_full.csv", index=False)
print("✅ Batch testing complete! Results saved to api_results_full.csv")
