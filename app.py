import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch_geometric.nn import ChebConv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import contextlib

# --- 1. DEFINE THE MODEL ARCHITECTURE (Must match Colab exactly) ---
class ChebyshevConvolutionLin(Module):
    def __init__(self, num_classes=2, kernel=[2, 2], num_features=165, hidden_units=512):
        super().__init__()
        self.conv1 = ChebConv(num_features, hidden_units, K=kernel[0])
        self.conv2 = ChebConv(hidden_units, hidden_units, K=kernel[1])
        self.linear = Linear(hidden_units, num_classes)

    def forward(self, x, edge_index):
        # Note: In production, for a single node, edge_index might be empty 
        # or contain self-loops. We handle this robustness below.
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        return x

# --- 2. GLOBAL VARIABLES ---
model = None
device = torch.device('cpu') # Use CPU for inference (cheaper/easier)

# --- 3. LIFESPAN MANAGER (Loads model on startup) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("🔄 Loading Model Weights...")
    
    # Initialize the Architecture
    model = ChebyshevConvolutionLin()
    
    # Load the Weights we trained in Colab
    try:
        model.load_state_dict(torch.load("production/weights/cheb_model_production.pth", map_location=device))
        model.to(device)
        model.eval() # Set to Evaluation Mode (Critical!)
        print("✅ Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ FAILED to load model: {e}")
        print("Did you place 'cheb_model_production.pth' in the 'production/weights' folder?")
    
    yield
    print("🛑 Shutting down API...")

# --- 4. INITIALIZE API ---
app = FastAPI(title="AML Bitcoin Detection API", lifespan=lifespan)

# --- 5. DEFINE INPUT FORMAT (Data Validation) ---
class TransactionInput(BaseModel):
    features: List[float] # Expecting a list of 165 numbers
    txId: str = "unknown_tx"

# --- 6. DEFINE ENDPOINTS ---
@app.get("/")
def home():
    return {"status": "online", "message": "Anti-Money Laundering GNN is ready."}

@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Prepare Data
    # Convert list to Tensor [1, 165]
    x = torch.tensor([transaction.features], dtype=torch.float).to(device)
    
    # For a single transaction isolated from the graph, we use a dummy self-loop edge
    # or empty edge_index depending on ChebConv requirements. 
    # ChebConv strictly requires edge_index. We create a self-loop (0->0).
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)

    # 2. Run Inference
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    # 3. Return Result
    label = "ILLICIT" if prediction == 1 else "LICIT"
    
    return {
        "txId": transaction.txId,
        "prediction": label,
        "confidence": f"{confidence:.4f}",
        "raw_probabilities": {
            "licit": f"{probs[0][0]:.4f}",
            "illicit": f"{probs[0][1]:.4f}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)