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
from classicArchitecture import ShallowRegressionLSTM, HybridClassicAttentionModel
from DataUtils import get_parkinsons_logo_folds



EXPERIMENTS = {
    # === MODELLI CLASSICI ===
    "experiment_classic_1": {
        "name": "Classic LSTM",
        "model_type": "Classic",
        "num_layer_tensor_ring": 0,
        "useAttention": False,
        "use_modified_ring": False,
    },
    "experiment_classic_2": {
        "name": "Classic LSTM with Attention",
        "model_type": "Classic",
        "num_layer_tensor_ring": 0,
        "useAttention": True,
        "use_modified_ring": False,
    },
    # === QLSTM STANDARD ===
    "experiment_1": {
        "name": "QLSTM standard",
        "model_type": "QLSTM",
        "num_layer_tensor_ring": 0,
        "useAttention": False,
        "use_modified_ring": False,
    },
    "experiment_2": {
        "name": "QLSTM standard with Attention",
        "model_type": "QLSTM",
        "num_layer_tensor_ring": 0,
        "useAttention": True,
        "use_modified_ring": False,
    },
    # === TENSOR RING STANDARD (use_modified_ring=False) ===
    "experiment_3": {
        "name": "QLSTM Tensor Ring 1 layer",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": False,
        "use_modified_ring": False,
    },
    "experiment_4": {
        "name": "QLSTM Tensor Ring 1 layer with Attention",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": True,
        "use_modified_ring": False,
    },
    "experiment_5": {
        "name": "QLSTM Tensor Ring 2 layers",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": False,
        "use_modified_ring": False,
    },
    "experiment_6": {
        "name": "QLSTM Tensor Ring 2 layers with Attention",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": True,
        "use_modified_ring": False,
    },
    # === TENSOR RING MODIFIED (use_modified_ring=True) ===
    "experiment_7": {
        "name": "QLSTM Tensor Ring Modified 1 layer",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": False,
        "use_modified_ring": True,
    },
    "experiment_8": {
        "name": "QLSTM Tensor Ring Modified 1 layer with Attention",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 1,
        "useAttention": True,
        "use_modified_ring": True,
    },
    "experiment_9": {
        "name": "QLSTM Tensor Ring Modified 2 layers",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": False,
        "use_modified_ring": True,
    },
    "experiment_10": {
        "name": "QLSTM Tensor Ring Modified 2 layers with Attention",
        "model_type": "TensorRing",
        "num_layer_tensor_ring": 2,
        "useAttention": True,
        "use_modified_ring": True,
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
    use_modified_ring = config.get("use_modified_ring", False)

    # === MODELLI CLASSICI ===
    if model_type == "Classic":
        if use_attention:
            return HybridClassicAttentionModel(
                total_input_features=total_features,
                hidden_dim=hidden_dim,
                num_layers=1,
                num_heads=1,
            )
        return ShallowRegressionLSTM(
            num_sensors=total_features,
            hidden_units=hidden_dim,
            num_layers=1,
        )

    # === MODELLI QUANTISTICI CON ATTENTION ===
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
            use_modified_ring=use_modified_ring,
        )

    # === MODELLI QUANTISTICI SENZA ATTENTION ===
    if model_type == "TensorRing":
        return QShallowRegressionLSTMTensorRing(
            num_sensors=total_features,
            hidden_units=hidden_dim,
            n_qubits=n_qubits,
            n_qlayers=max(1, num_layer_tensor_ring),
            use_modified_ring=use_modified_ring,
        )

    return QShallowRegressionLSTM(
        num_sensors=total_features,
        hidden_units=hidden_dim,
        n_qubits=n_qubits,
        n_qlayers=default_n_qlayers,
    )

# --- CONFIGURAZIONE & CARICAMENTO DATI ---

print("Inizializzazione esperimenti con Leave-One-Group-Out Cross-Validation...")

# Parametri Dataset
window_size = 10
step = 1

# Parametri Modello
hidden_dim = 4
n_qubits = 4
n_qlayers = 1
epochs = 100
learning_rate = 0.001

results = []

# --- LEAVE ONE GROUP OUT CROSS-VALIDATION ---
# DataUtils.py gestisce tutto: caricamento, windowing, normalizzazione per fold
print(f"\n=== Inizio Leave-One-Group-Out Cross-Validation ===")

for fold_data in get_parkinsons_logo_folds(window_size=window_size, step=step):
    fold_idx = fold_data["fold_idx"]
    test_patient = fold_data["test_patient"]
    feature_names = fold_data["feature_names"]
    n_folds = fold_data["n_folds"]
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{n_folds} - Test su paziente {test_patient}")
    print(f"Train: {fold_data['n_train']} campioni, Test: {fold_data['n_test']} campioni")
    print(f"{'='*60}")
    
    # Dati già normalizzati e pronti all'uso
    X_train = torch.tensor(fold_data["X_train"], dtype=torch.float32)
    X_test = torch.tensor(fold_data["X_test"], dtype=torch.float32)
    y_train = torch.tensor(fold_data["y_train"], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(fold_data["y_test"], dtype=torch.float32).unsqueeze(1)
    
    # Determina il numero di feature dalla prima iterazione
    if fold_idx == 0:
        total_features = X_train.shape[2]
        print(f"Feature utilizzate ({total_features}): {feature_names}")
    
    # Eseguo ogni esperimento su questo fold
    for exp_id, config in EXPERIMENTS.items():
        print(f"\n--- Esperimento: {config['name']} ---")
        
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
            "fold": fold_idx + 1,
            "test_patient": test_patient,
            "name": config["name"],
            "modelType": config["model_type"],
            "num_layer_tensor_ring": config["num_layer_tensor_ring"],
            "useAttention": config["useAttention"],
            "use_modified_ring": config.get("use_modified_ring", False),
            **metrics,
        })
        
        print(f"Metriche - MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}, "
              f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")

# --- 3. SALVATAGGIO RISULTATI ---
output_path = os.path.join(os.path.dirname(__file__), "experiment_results.csv")
fieldnames = [
    "fold",
    "test_patient",
    "name",
    "modelType",
    "num_layer_tensor_ring",
    "useAttention",
    "use_modified_ring",
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

print(f"\n{'='*60}")
print(f"Risultati salvati in: {output_path}")
print(f"Totale esperimenti eseguiti: {len(results)}")
print(f"{'='*60}")