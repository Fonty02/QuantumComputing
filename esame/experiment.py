import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hybridArchitecture import HybridQuantumAttentionModel
from QLSTM import QShallowRegressionLSTM
from QLSTMTensorRing import QShallowRegressionLSTMTensorRing
from DataUtils import get_traffic_dataset



EXPERIMENTS = {
    "experiment_1": {
        "name":"QLSTM standard",
        "model_type":"QLSTM",
        "num_layer_tensor_ring":0, #since we are not using tensor ring
        "useAttention":False
    },
    "experiment_2": {
        "name":"QLSTM standard with Attention",
        "model_type":"QLSTM",
        "num_layer_tensor_ring":0, #since we are not using tensor ring
        "useAttention":True
    },
    "experiment_3": {
        "name":"QLSTM Tensor Ring 1 layer",
        "model_type":"TensorRing",
        "num_layer_tensor_ring":1, #number of tensor ring layers
        "useAttention":False
    },
    "experiment_4": {
        "name":"QLSTM Tensor Ring 1 layer with Attention",
        "model_type":"TensorRing",
        "num_layer_tensor_ring":1, #number of tensor ring layers
        "useAttention":True
    },
    "experiment_5": {
        "name":"QLSTM Tensor Ring 2 layers",
        "model_type":"TensorRing",
        "num_layer_tensor_ring":2, #number of tensor ring layers
        "useAttention":False
    },
    "experiment_6": {
        "name":"QLSTM Tensor Ring 2 layers with Attention",
        "model_type":"TensorRing",
        "num_layer_tensor_ring":2, #number of tensor ring layers
        "useAttention":True
    },
}


def train_model(model, X_train, y_train, epochs=20, lr=0.01, name="Model"):
    """
    Funzione generica per addestrare un modello PyTorch
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    
    print(f"\n--- Inizio Training: {name} ---")
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        # X_train shape: (Batch, Seq, Features)
        outputs = model(X_train)
        
        # Loss calculation
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
    elapsed = time.time() - start_time
    print(f"Training completato in {elapsed:.2f} secondi.")
    return losses, model


def evaluate_regression(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)
    mape = float((abs((y_true_np - y_pred_np) / (y_true_np + 1e-8))).mean()) * 100.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
    }


def time_series_split(X, y, test_ratio=0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def build_model(
    config,
    total_features,
    hidden_dim,
    n_qubits,
    default_n_qlayers,
    backend,
):
    model_type = config["model_type"]
    use_attention = config["useAttention"]
    num_layer_tensor_ring = config["num_layer_tensor_ring"]

    if use_attention:
        n_qlayers = default_n_qlayers
        if model_type == "TensorRing":
            n_qlayers = max(1, num_layer_tensor_ring)

        return HybridQuantumAttentionModel(
            total_input_features=total_features,
            hidden_dim=hidden_dim,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            model_type=model_type,
            backend=backend,
        )

    if model_type == "TensorRing":
        return QShallowRegressionLSTMTensorRing(
            num_sensors=total_features,
            hidden_units=hidden_dim,
            n_qubits=n_qubits,
            n_qlayers=max(1, num_layer_tensor_ring),
        )

    return QShallowRegressionLSTM(
        num_sensors=total_features,
        hidden_units=hidden_dim,
        n_qubits=n_qubits,
        n_qlayers=default_n_qlayers,
    )

# --- CONFIGURAZIONE & CARICAMENTO DATI ---

# 1. Carichiamo i dati (Usa la funzione get_traffic_dataset definita prima)
# Usiamo un subset piccolo (200 campioni) per velocità, dato che è una simulazione quantistica
print("Caricamento Dataset...")
X_all, y_all, scaler = get_traffic_dataset(window_size=4, train_samples=300)

# Parametri Modello
total_features = X_all.shape[2]  # Target + Covariate
hidden_dim = 4
n_qubits = 4
n_qlayers = 1
epochs = 100
learning_rate = 0.001

# --- 2. SPLIT TRAIN/TEST ---
X_train, X_test, y_train, y_test = time_series_split(X_all, y_all, test_ratio=0.3)

results = []

for exp_id, config in EXPERIMENTS.items():
    model = build_model(
        config=config,
        total_features=total_features,
        hidden_dim=hidden_dim,
        n_qubits=n_qubits,
        default_n_qlayers=n_qlayers,
        backend="default.qubit",
    )

    losses, model = train_model(
        model,
        X_train,
        y_train,
        epochs=epochs,
        lr=learning_rate,
        name=config["name"],
    )

    model.eval()
    with torch.no_grad():
        preds = model(X_test)

    metrics = evaluate_regression(y_test, preds)

    results.append({
        "name": config["name"],
        "modelType": config["model_type"],
        "num_layer_tensor_ring": config["num_layer_tensor_ring"],
        "useAttention": config["useAttention"],
        **metrics,
    })

# --- 3. SALVATAGGIO RISULTATI ---
output_path = os.path.join(os.path.dirname(__file__), "experiment_results.csv")
fieldnames = [
    "name",
    "modelType",
    "num_layer_tensor_ring",
    "useAttention",
    "mse",
    "rmse",
    "mae",
    "r2",
    "mape",
]

with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Risultati salvati in: {output_path}")