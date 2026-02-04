import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data
from torch.optim import Adam
from sklearn.metrics import f1_score, recall_score

# --- 1. CONFIGURATION ---
ARGS = {
    'lr': 0.01,
    'weight_decay': 0.0005,
    'epochs': 1000,
    'hidden_units': 512,
    'num_classes': 2,
    'num_hops': [2, 2],  # K=2
    'seed': 42,
    # Paths (Relative to the root where you run the command)
    'data_path': 'data/',
    'save_path': 'production/weights/cheb_model_production.pth'
}

# --- 2. REPRODUCIBILITY ---
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed}")

# --- 3. DATA LOADING ---
def load_data_production(data_path):
    print("🔄 Loading Elliptic Dataset...")
    
    # Check if files exist
    if not os.path.exists(os.path.join(data_path, "elliptic_txs_features.csv")):
        raise FileNotFoundError(f"❌ Data not found in {data_path}. Please check your folder structure.")

    # 1. READ DATA
    df_edges = pd.read_csv(os.path.join(data_path, "elliptic_txs_edgelist.csv"))
    df_features = pd.read_csv(os.path.join(data_path, "elliptic_txs_features.csv"), header=None)
    df_classes = pd.read_csv(os.path.join(data_path, "elliptic_txs_classes.csv"))

    # 2. RENAME COLUMNS (Standardizing feature names)
    colNames1 = {'0': 'txId', 1: "time_step"}
    colNames2 = {str(ii+2): "Local_feature_" + str(ii+1) for ii in range(94)}
    colNames3 = {str(ii+96): "Aggregate_feature_" + str(ii+1) for ii in range(72)}
    colNames = dict(colNames1, **colNames2, **colNames3)
    colNames = {int(jj): item_kk for jj, item_kk in colNames.items()}
    df_features = df_features.rename(columns=colNames)

    # 3. DROP UNKNOWNS & MAP CLASSES
    # 'unknown' -> 3 (we drop these), '1' (Illicit) -> 1, '2' (Licit) -> 0
    df_classes['class'] = df_classes['class'].replace({'unknown': 3, '1': 1, '2': 0})

    # Merge
    df_merged = pd.merge(df_classes, df_features, on="txId", how='inner')
    df_merged = df_merged[df_merged['class'] != 3] # Drop unknown

    # 4. MAP TXID TO INTEGERS
    nodes = df_merged['txId'].unique()
    map_id = {j: i for i, j in enumerate(nodes)}
    df_merged['txId_mapped'] = df_merged['txId'].map(map_id)

    # Map Edges
    df_edges = df_edges[df_edges['txId1'].isin(nodes) & df_edges['txId2'].isin(nodes)]
    df_edges['source'] = df_edges['txId1'].map(map_id)
    df_edges['target'] = df_edges['txId2'].map(map_id)

    # 5. CONVERT TO PYTORCH
    edge_index = torch.tensor(df_edges[['source', 'target']].values.T, dtype=torch.long)
    
    drop_cols = ['txId', 'class', 'time_step', 'txId_mapped']
    existing_drop = [c for c in drop_cols if c in df_merged.columns]
    x = torch.tensor(df_merged.drop(existing_drop, axis=1).values, dtype=torch.float)
    y = torch.tensor(df_merged['class'].values, dtype=torch.long)

    # 6. TEMPORAL SPLIT (Train < 35)
    time_step = df_merged['time_step'].values
    train_mask = time_step < 35
    test_mask = time_step >= 35

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = torch.tensor(train_mask)
    data.test_mask = torch.tensor(test_mask)

    print(f"✅ Data Ready! Nodes: {data.num_nodes}, Features: {data.num_features}")
    return data

# --- 4. MODEL DEFINITION (MUST MATCH APP.PY) ---
class ChebyshevConvolutionLin(Module):
    def __init__(self, args, kernel, num_features, hidden_units):
        super().__init__()
        self.conv1 = ChebConv(num_features, hidden_units, K=kernel[0])
        self.conv2 = ChebConv(hidden_units, hidden_units, K=kernel[1])
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=0.5)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        
        x = self.linear(x)
        return x, edge_index

# --- 5. TRAINING LOOP ---
def train_model():
    seed_everything(ARGS['seed'])
    
    # Load Data
    data = load_data_production(ARGS['data_path'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Initialize Model
    model = ChebyshevConvolutionLin(
        ARGS,
        kernel=ARGS['num_hops'],
        num_features=data.num_features,
        hidden_units=ARGS['hidden_units']
    ).to(device)
    
    # Optimizer & Loss (Weighted for Imbalance)
    optimizer = Adam(model.parameters(), lr=ARGS['lr'], weight_decay=ARGS['weight_decay'])
    class_weights = torch.tensor([1.0, 3.0]).to(device) # Penalty for missing Illicit (Class 1)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"🚀 Starting training on {device}...")
    
    for epoch in range(ARGS['epochs'] + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            pred = out.argmax(dim=1)
            f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), pos_label=1, zero_division=0)
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Test F1 (Illicit): {f1:.4f}")

    # --- 6. SAVE MODEL ---
    # Create directory if it doesn't exist (Safety Check)
    save_dir = os.path.dirname(ARGS['save_path'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📂 Created directory: {save_dir}")

    torch.save(model.state_dict(), ARGS['save_path'])
    print(f"✅ Model successfully saved to: {ARGS['save_path']}")

if __name__ == "__main__":
    train_model()