import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hybridArchitecture import HybridQuantumAttentionModel
from QLSTM import QShallowRegressionLSTM
from QLSTMTensorRing import QShallowRegressionLSTMTensorRing
from DataUtilsTraffic import get_traffic_dataset
from expressivity import compute_expressivity_from_model



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
    }
}
"""
        ,
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
    """



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


def set_seed(seed=42):
    """
    Set all random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All seeds set to {seed} for reproducibility")


def compute_model_expressivity(model, config, n_samples=1000, seed=42):
    """
    Compute expressivity (KL-divergence) for a trained model.
    
    This function uses the actual circuit structure from the trained model,
    ensuring the expressivity is calculated on the exact same circuit architecture.
    
    Following the methodology from the slides:
    - Lower KL-divergence = higher expressivity = circuit covers more of Hilbert space
    - Uses Haar distribution as the reference (ideal random distribution)
    
    Args:
        model: Trained model (QShallowRegressionLSTM or QShallowRegressionLSTMTensorRing)
        config: Experiment configuration dictionary
        n_samples: Number of random parameter samples
        seed: Random seed for reproducibility
        
    Returns:
        float: KL-divergence value (lower = more expressive)
    """
    model_type = config["model_type"]
    
    # Classic models don't have quantum expressivity
    if model_type == "Classic":
        return None
    
    try:
        # Use the model's actual circuit structure
        kl_divergence, _, _ = compute_expressivity_from_model(
            model=model,
            gate_name='forget',  # Use forget gate as representative
            n_samples=n_samples,
            seed=seed
        )
        return kl_divergence
    except Exception as e:
        print(f"Warning: Could not compute expressivity: {e}")
        return None


# --- CONFIGURATION & DATA LOADING ---

# Set seed for reproducibility
SEED = 42
set_seed(SEED)

# Model Parameters
hidden_dim = 4
n_qubits = 4
n_qlayers = 1
epochs = 500
learning_rate = 0.001
patience = 5
train_samples = 24102
test_ratio = 0.3

# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

# Create plot directory if it doesn't exist
plot_dir = os.path.join(os.path.dirname(__file__), "plot")
os.makedirs(plot_dir, exist_ok=True)

# Prepare CSV file with header
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
    "expressivity_kl",
]

# Create CSV with header if it doesn't exist
if not os.path.exists(output_path):
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    print(f"Created new results file: {output_path}")
else:
    print(f"Appending to existing results file: {output_path}")

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
    
    # Check if model already exists
    model_filename = f"{exp_id}.pth"
    model_path = os.path.join(models_dir, model_filename)
    
    model = build_model(
        config=config,
        total_features=total_features,
        hidden_dim=hidden_dim,
        n_qubits=n_qubits,
        default_n_qlayers=n_qlayers,
        backend="default.qubit",
    )
    
    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        print("Loading existing model and skipping training...")
        model.load_state_dict(torch.load(model_path))
        used_epochs = 0  # No training performed
    else:
        print("Training new model...")
        losses, model, used_epochs = train_model(
            model,
            X_train,
            y_train,
            epochs=epochs,
            lr=learning_rate,
            name=config["name"],
            patience=patience,
        )
        
        # Save the best model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test)

    metrics = evaluate_regression(y_test, preds)
    
    # Compute expressivity for quantum models
    print("Computing expressivity (KL-divergence w.r.t. Haar distribution)...")
    expressivity_kl = compute_model_expressivity(
        model=model,
        config=config,
        n_samples=500,  # Reduced samples for speed, increase for more accuracy
        seed=SEED
    )
    if expressivity_kl is not None:
        print(f"Expressivity (KL-divergence): {expressivity_kl:.6f} (lower = more expressive)")
    else:
        print("Expressivity: N/A (Classic model)")
    
    # Create prediction comparison plot
    plt.figure(figsize=(12, 6))
    y_test_np = y_test.detach().cpu().numpy().flatten()
    preds_np = preds.detach().cpu().numpy().flatten()
    
    plt.plot(y_test_np, label='Ground Truth', linewidth=2, alpha=0.8)
    plt.plot(preds_np, label='Predictions', linewidth=2, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic Volume')
    plt.title(f'{config["name"]} - Prediction vs Ground Truth (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{exp_id}_predictions.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")

    # Prepare result row
    result_row = {
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
        "expressivity_kl": expressivity_kl,
    }
    
    # Append result to CSV immediately
    with open(output_path, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(result_row)
    
    print(f"Result appended to: {output_path}")

print(f"\n{'='*80}")
print(f"All experiments completed!")
print(f"Results saved to: {output_path}")
print(f"{'='*80}")