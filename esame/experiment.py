import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from hybridArchitecture import HybridQuantumAttentionModel
from DataUtils import get_traffic_dataset


def train_model(model, X_train, y_train, epochs=20, lr=0.01, name="Model"):
    """
    Funzione generica per addestrare un modello PyTorch
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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

# --- CONFIGURAZIONE & CARICAMENTO DATI ---

# 1. Carichiamo i dati (Usa la funzione get_traffic_dataset definita prima)
# Usiamo un subset piccolo (200 campioni) per velocità, dato che è una simulazione quantistica
print("Caricamento Dataset...")
X_train, y_train, scaler = get_traffic_dataset(window_size=4, train_samples=200)

# Parametri Modello
total_features = X_train.shape[2] # Target + Covariate
hidden_dim = 4
n_qubits = 4
n_qlayers = 1
epochs = 1000
learning_rate = 0.05

# --- 2. TRAINING MODELLO 1: STANDARD QLSTM ---
model_standard = HybridQuantumAttentionModel(
    total_input_features=total_features,
    hidden_dim=hidden_dim,
    n_qubits=n_qubits,
    n_qlayers=n_qlayers,
    model_type='QLSTM',       # <--- Seleziona Standard
    backend='default.qubit'
)

losses_standard, model_standard = train_model(
    model_standard, X_train, y_train, epochs=epochs, lr=learning_rate, name="Standard QLSTM"
)

# --- 3. TRAINING MODELLO 2: TENSOR RING QLSTM ---
model_tensor = HybridQuantumAttentionModel(
    total_input_features=total_features,
    hidden_dim=hidden_dim,
    n_qubits=n_qubits,
    n_qlayers=n_qlayers,
    model_type='TensorRing',  # <--- Seleziona Tensor Ring
    backend='default.qubit'
)

losses_tensor, model_tensor = train_model(
    model_tensor, X_train, y_train, epochs=epochs, lr=learning_rate, name="TensorRing QLSTM"
)

# --- 4. VISUALIZZAZIONE RISULTATI ---

# Plot Loss Comparison
plt.figure(figsize=(10, 5))
plt.plot(losses_standard, label='Standard QLSTM Loss', linestyle='--')
plt.plot(losses_tensor, label='TensorRing QLSTM Loss')
plt.title('Confronto Loss durante il Training')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_comparison.png')
plt.show()

# Plot Predictions (sul Training Set per verifica)
model_standard.eval()
model_tensor.eval()
with torch.no_grad():
    pred_standard = model_standard(X_train).numpy()
    pred_tensor = model_tensor(X_train).numpy()
    actual = y_train.numpy()

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual Data', color='black', alpha=0.6)
plt.plot(pred_standard, label='Standard QLSTM Pred', linestyle='--')
plt.plot(pred_tensor, label='TensorRing QLSTM Pred')
plt.title('Confronto Predizioni (Training Set)')
plt.legend()
plt.savefig('prediction_comparison.png')
plt.show()