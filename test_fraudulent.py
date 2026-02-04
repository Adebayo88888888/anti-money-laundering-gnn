import requests
import pandas as pd
import json

# 1. Load the Data
print("⏳ Loading dataset files (this might take some seconds)...")
classes_df = pd.read_csv('data/elliptic_txs_classes.csv')
features_df = pd.read_csv('data/elliptic_txs_features.csv', header=None)

# 2. Find a "Bad Guy" (Class 1 = Illicit)
# The dataset labels are: '1' (Illicit), '2' (Licit), 'unknown'
illicit_txs = classes_df[classes_df['class'] == '1']
target_tx_id = illicit_txs.iloc[0]['txId'] # Grab the first one we find

print(f"🎯 Hunting for Illicit Transaction ID: {target_tx_id}")

# 3. Get the features for this specific Bad Transaction
# We find the row in the features file that matches the ID
target_row = features_df[features_df[0] == target_tx_id]
real_features = target_row.iloc[0, 2:167].values.tolist()

# 4. Send to API
payload = {
    "features": real_features,
    "txId": str(target_tx_id)
}

print("👮‍♀️ Sending suspicious transaction to API...")
url = "http://127.0.0.1:8000/predict"

try:
    response = requests.post(url, json=payload)
    print("\n🔎 MODEL VERDICT:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Failed to connect: {e}")