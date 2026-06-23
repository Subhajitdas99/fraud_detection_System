import pandas as pd

# Load the full credit card dataset
df = pd.read_csv("creditcard.csv")

# Number of test samples to create
n_samples = 1000  

# Take a random sample
test_df = df.sample(n_samples, random_state=42)

# Save as test CSV
test_df.to_csv("test_transactions_full.csv", index=False)

print(f"✅ Created test_transactions_full.csv with {n_samples} sampled rows from creditcard.csv")

