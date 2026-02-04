import requests
import pandas as pd
import json

# 1. Load the dataset (Just the first 5 rows to save memory)
print("⏳ Loading a real transaction from CSV...")
# We use header=None because the Elliptic dataset has no column names
df = pd.read_csv('data/elliptic_txs_features.csv', header=None, nrows=5)

# 2. Pick a transaction (Row 0)
# Column 0 is txId. Column 1 is Time step. Features are Columns 2 to 166.
real_tx_id = str(df.iloc[0, 0])
real_features = df.iloc[0, 2:167].values.tolist() # columns 2 to 166 = 165 features

print(f"🔹 Testing Transaction ID: {real_tx_id}")
print(f"🔹 Feature Count: {len(real_features)}")

# 3. Define the payload
payload = {
    "features": real_features,
    "txId": real_tx_id
}

# 4. Send to API
url = "http://127.0.0.1:8000/predict"

try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("\n✅ API PREDICTION RECEIVED:")
        # Pretty print the JSON result
        print(json.dumps(response.json(), indent=2))
    else:
        print("❌ Error:", response.text)
except Exception as e:
    print(f"Failed to connect: {e}")