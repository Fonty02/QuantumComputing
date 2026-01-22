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
from DataUtilsTraffic import get_traffic_dataset



EXPERIMENTS = {
    # Classic models - Window Size 4
    "experiment_classic_1_ws4": {
        "name": "Classic LSTM (WS=4)",
        "model_type": "Classic",
        "num_layer_tensor_ring": 0,
        "useAttention": False,
        "use_modified_ring": False,
        "window_size": 4,
    },
    "experiment_classic_2_ws4": {
        "name": "Classic LSTM with Attention (WS=4)",
        "model_type": "Classic",
        "num_layer_tensor_ring": 0,
        "useAttention": True,
        "use_modified_ring": False,
        "window_size": 4,
    },
    # Standard QLSTM - Window Size 4
    "experiment_1_ws4": {
        "name": "QLSTM standard (WS=4)",
        "model_type": "QLSTM",
        "num_layer_tensor_ring": 0,
        "useAttention": False,
        "use_modified_ring": False,
        "window_size": 4,
    },
    "experiment_2_ws4": {
        "name": "QLSTM standard with Attention (WS=4)",
        "model_type": "QLSTM",
        "num_layer_tensor_ring": 0,
        "useAttention": True,
        "use_modified_ring": False,
        "window_size": 4,
    },
    # Tensor Ring Standard - Window Size 4
    "experiment_3_ws4": {
        "name": "QLSTM Tensor Ring 1 layer (WS=4)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": False,
        "use_modified_ring": False,
        "window_size": 4,
    },
    "experiment_4_ws4": {
        "name": "QLSTM Tensor Ring 1 layer with Attention (WS=4)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": True,
        "use_modified_ring": False,
        "window_size": 4,
    },
    "experiment_5_ws4": {
        "name": "QLSTM Tensor Ring 2 layers (WS=4)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": False,
        "use_modified_ring": False,
        "window_size": 4,
    },
    "experiment_6_ws4": {
        "name": "QLSTM Tensor Ring 2 layers with Attention (WS=4)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": True,
        "use_modified_ring": False,
        "window_size": 4,
    },
    # Tensor Ring Modified - Window Size 4
    "experiment_7_ws4": {
        "name": "QLSTM Tensor Ring Modified 1 layer (WS=4)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": False,
        "use_modified_ring": True,
        "window_size": 4,
    },
    "experiment_8_ws4": {
        "name": "QLSTM Tensor Ring Modified 1 layer with Attention (WS=4)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": True,
        "use_modified_ring": True,
        "window_size": 4,
    },
    "experiment_9_ws4": {
        "name": "QLSTM Tensor Ring Modified 2 layers (WS=4)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": False,
        "use_modified_ring": True,
        "window_size": 4,
    },
    "experiment_10_ws4": {
        "name": "QLSTM Tensor Ring Modified 2 layers with Attention (WS=4)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": True,
        "use_modified_ring": True,
        "window_size": 4,
    },
    
    # Classic models - Window Size 72
    "experiment_classic_1_ws72": {
        "name": "Classic LSTM (WS=72)",
        "model_type": "Classic",
        "num_layer_tensor_ring": 0,
        "useAttention": False,
        "use_modified_ring": False,
        "window_size": 72,
    },
    "experiment_classic_2_ws72": {
        "name": "Classic LSTM with Attention (WS=72)",
        "model_type": "Classic",
        "num_layer_tensor_ring": 0,
        "useAttention": True,
        "use_modified_ring": False,
        "window_size": 72,
    },
    # Standard QLSTM - Window Size 72
    "experiment_1_ws72": {
        "name": "QLSTM standard (WS=72)",
        "model_type": "QLSTM",
        "num_layer_tensor_ring": 0,
        "useAttention": False,
        "use_modified_ring": False,
        "window_size": 72,
    },
    "experiment_2_ws72": {
        "name": "QLSTM standard with Attention (WS=72)",
        "model_type": "QLSTM",
        "num_layer_tensor_ring": 0,
        "useAttention": True,
        "use_modified_ring": False,
        "window_size": 72,
    },
    # Tensor Ring Standard - Window Size 72
    "experiment_3_ws72": {
        "name": "QLSTM Tensor Ring 1 layer (WS=72)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": False,
        "use_modified_ring": False,
        "window_size": 72,
    },
    "experiment_4_ws72": {
        "name": "QLSTM Tensor Ring 1 layer with Attention (WS=72)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": True,
        "use_modified_ring": False,
        "window_size": 72,
    },
    "experiment_5_ws72": {
        "name": "QLSTM Tensor Ring 2 layers (WS=72)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": False,
        "use_modified_ring": False,
        "window_size": 72,
    },
    "experiment_6_ws72": {
        "name": "QLSTM Tensor Ring 2 layers with Attention (WS=72)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": True,
        "use_modified_ring": False,
        "window_size": 72,
    },
    # Tensor Ring Modified - Window Size 72
    "experiment_7_ws72": {
        "name": "QLSTM Tensor Ring Modified 1 layer (WS=72)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": False,
        "use_modified_ring": True,
        "window_size": 72,
    },
    "experiment_8_ws72": {
        "name": "QLSTM Tensor Ring Modified 1 layer with Attention (WS=72)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": True,
        "use_modified_ring": True,
        "window_size": 72,
    },
    "experiment_9_ws72": {
        "name": "QLSTM Tensor Ring Modified 2 layers (WS=72)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": False,
        "use_modified_ring": True,
        "window_size": 72,
    },
    "experiment_10_ws72": {
        "name": "QLSTM Tensor Ring Modified 2 layers with Attention (WS=72)",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": True,
        "use_modified_ring": True,
        "window_size": 72,
    },
}


def train_model(model, X_train, y_train, epochs=20, lr=0.01, name="Model", patience=50):
    """
    Generic function to train a PyTorch model with early stopping
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\n--- Training Start: {name} ---")
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
        
        current_loss = loss.item()
        losses.append(current_loss)
        
        # Early stopping logic
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {current_loss:.6f}, Best: {best_loss:.6f}, Patience: {patience_counter}/{patience}")
        
        # Check early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            elapsed = time.time() - start_time
            print(f"Training completed in {elapsed:.2f} seconds.")
            return losses, model, epoch + 1  # Return actual number of epochs used
            
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")
    return losses, model, epochs  # Return total epochs if no early stopping


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

# --- CONFIGURATION & DATA LOADING ---

# Model Parameters
hidden_dim = 4
n_qubits = 4
n_qlayers = 1
epochs = 500
learning_rate = 0.001
patience = 5
train_samples = 24102
test_ratio = 0.3

results = []

# Pre-load datasets for both window sizes
datasets_cache = {}
for ws in [4, 72]:
    print(f"\n{'='*80}")
    print(f"Loading Dataset with window_size={ws}...")
    print(f"{'='*80}")
    X_all, y_all, scaler = get_traffic_dataset(window_size=ws, train_samples=train_samples)
    X_train, X_test, y_train, y_test = time_series_split(X_all, y_all, test_ratio=test_ratio)
    datasets_cache[ws] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "total_features": X_all.shape[2],
    }

# Run all experiments
for exp_id, config in EXPERIMENTS.items():
    window_size = config["window_size"]
    
    print(f"\n{'*'*60}")
    print(f"Experiment: {config['name']}")
    print(f"{'*'*60}")
    
    # Get cached data for this window size
    data = datasets_cache[window_size]
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    total_features = data["total_features"]
    
    model = build_model(
        config=config,
        total_features=total_features,
        hidden_dim=hidden_dim,
        n_qubits=n_qubits,
        default_n_qlayers=n_qlayers,
        backend="default.qubit",
    )

    losses, model, used_epochs = train_model(
        model,
        X_train,
        y_train,
        epochs=epochs,
        lr=learning_rate,
        name=config["name"],
        patience=patience,
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
        "use_modified_ring": config["use_modified_ring"],
        "window_size": window_size,
        "total_epochs": epochs,
        "used_epochs": used_epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "n_qubits": n_qubits,
        "n_qlayers": n_qlayers if config["model_type"] != "TensorRing" else max(1, config["num_layer_tensor_ring"]),
        "test_ratio": test_ratio,
        "total_features": total_features,
        **metrics,
    })

# --- 3. SAVE RESULTS ---
output_path = os.path.join(os.path.dirname(__file__), "experiment_results.csv")
fieldnames = [
    "name",
    "modelType",
    "num_layer_tensor_ring",
    "useAttention",
    "use_modified_ring",
    "window_size",
    "total_epochs",
    "used_epochs",
    "patience",
    "learning_rate",
    "hidden_dim",
    "n_qubits",
    "n_qlayers",
    "test_ratio",
    "total_features",
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

print(f"\n{'='*80}")
print(f"Results saved to: {output_path}")
print(f"Total experiments completed: {len(results)}")
print(f"{'='*80}")