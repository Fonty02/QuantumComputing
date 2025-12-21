import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
from torch.utils.data import Dataset, DataLoader
import requests
import io
from tqdm import tqdm


# ==================== QUANTUM CIRCUIT ====================
class QuantumCircuit:
    """Variational Quantum Circuit per QGRU"""
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Usa lightning.qubit per simulazione più veloce se disponibile
        try:
            self.dev = qml.device('lightning.qubit', wires=n_qubits)
            print(f"✓ Usando lightning.qubit (veloce)")
        except:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            print(f"⚠ Usando default.qubit (lento)")
        
        # Rimuovo diff_method per lasciare che PennyLane scelga automaticamente
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # Angle Encoding
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            
            # Basic Entangler layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i], wires=i)
                
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 1:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.circuit = circuit
    
    def forward(self, inputs, weights):
        batch_size = inputs.shape[0]
        results = []
        
        for i in range(batch_size):
            circuit_output = self.circuit(inputs[i], weights)
            if isinstance(circuit_output, list):
                circuit_output = torch.stack(circuit_output)
            results.append(circuit_output.float())
        
        return torch.stack(results)


class VQCLayer(nn.Module):
    """Layer quantistico: FC -> VQC -> FC"""
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers):
        super(VQCLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        self.fc_in = nn.Linear(input_dim, n_qubits)
        self.qc = QuantumCircuit(n_qubits, n_layers)
        self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.fc_out = nn.Linear(n_qubits, hidden_dim)
    
    def forward(self, x):
        x = self.fc_in(x)
        x = torch.tanh(x)
        x = self.qc.forward(x, self.q_params)
        x = self.fc_out(x)
        return x


class QGRUCell(nn.Module):
    """Cella QGRU con tre VQC"""
    def __init__(self, input_dim, hidden_dim, n_qubits=5, n_layers=2):
        super(QGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        concat_dim = hidden_dim + input_dim
        
        self.vqc_reset = VQCLayer(concat_dim, hidden_dim, n_qubits, n_layers)
        self.vqc_update = VQCLayer(concat_dim, hidden_dim, n_qubits, n_layers)
        self.vqc_output = VQCLayer(concat_dim, hidden_dim, n_qubits, n_layers)
    
    def forward(self, x, h_prev):
        combined = torch.cat([h_prev, x], dim=1)
        
        r_t = torch.sigmoid(self.vqc_reset(combined))
        z_t = torch.sigmoid(self.vqc_update(combined))
        
        combined_reset = torch.cat([r_t * h_prev, x], dim=1)
        h_tilde = torch.tanh(self.vqc_output(combined_reset))
        
        h = z_t * h_prev + (1 - z_t) * h_tilde
        
        return h


class QGRU(nn.Module):
    """Quantum GRU completo"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_qubits=5, n_layers=2):
        super(QGRU, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.qgru_cell = QGRUCell(input_dim, hidden_dim, n_qubits, n_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h_0=None):
        batch_size, seq_len, _ = x.shape
        
        if h_0 is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h = h_0
        
        outputs = []
        for t in range(seq_len):
            h = self.qgru_cell(x[:, t, :], h)
            out = self.fc_out(h)
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, h


# ==================== CLASSICAL GRU ====================
class ClassicalGRU(nn.Module):
    """GRU classico per confronto"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(ClassicalGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h_0=None):
        out, h = self.gru(x, h_0)
        out = self.fc_out(out)
        return out, h


# ==================== DATASET ====================
class TimeSeriesDataset(Dataset):
    """Dataset per time series con finestre scorrevoli"""
    def __init__(self, data, window_size=5):
        self.data = data
        self.window_size = window_size
    
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size, :-1]  # features
        y = self.data[idx + 1:idx + self.window_size + 1, -1:]  # target (shifted by 1)
        return torch.FloatTensor(x), torch.FloatTensor(y)


def load_ettm1_data():
    """Carica il dataset ETTh1 (o genera dati sintetici se non disponibile)"""
    try:
        # Prova a scaricare ETTh1
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
        response = requests.get(url, timeout=10)
        df = pd.read_csv(io.StringIO(response.text))
        
        # Usa le prime 4 colonne (OT è il target) - DATASET COMPLETO
        data = df[['HUFL', 'HULL', 'MUFL', 'OT']].values
        print(f"✓ Dataset ETTh1 caricato con successo! ({len(data)} samples)")
        
    except Exception as e:
        print(f"⚠ Impossibile scaricare ETTh1: {e}")
        print("Generazione dati sintetici...")
        
        # Genera dati sintetici multi-variate - COMPLETO
        np.random.seed(42)
        t = np.linspace(0, 100, 5000)  # Dataset completo
        
        # 3 features + 1 target con pattern realistici
        data = np.column_stack([
            np.sin(t * 0.1) + np.random.normal(0, 0.1, len(t)),
            np.cos(t * 0.15) + np.random.normal(0, 0.1, len(t)),
            np.sin(t * 0.05) * np.cos(t * 0.1) + np.random.normal(0, 0.1, len(t)),
            np.sin(t * 0.1) + 0.5 * np.cos(t * 0.2) + np.random.normal(0, 0.15, len(t))
        ])
        print(f"✓ Dataset sintetico generato! ({len(data)} samples)")
    
    return data


# ==================== TRAINING ====================
def train_model(model, train_loader, val_loader, epochs, lr, device):
    """Training loop con metriche"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    epoch_times = []
    
    print(f"\nTraining su {device}...")
    print(f"Batches per epoca: {len(train_loader)}")
    
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False, unit="batch")
        for batch_x, batch_y in batch_progress:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_progress.set_postfix({"loss": f"{loss.item():.6f}"})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, _ = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # Optional: print epoch details if needed
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.6f} - "
              f"Val Loss: {val_loss:.6f} - "
              f"Time: {epoch_time:.2f}s")
    
    return train_losses, val_losses, epoch_times


def evaluate_model(model, test_loader, device):
    """Valutazione finale"""
    model.eval()
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    
    test_loss = 0
    test_mae = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, _ = model(batch_x)
            
            test_loss += criterion(outputs, batch_y).item()
            test_mae += mae_criterion(outputs, batch_y).item()
            
            all_preds.append(outputs.cpu())
            all_targets.append(batch_y.cpu())
    
    test_loss /= len(test_loader)
    test_mae /= len(test_loader)
    
    return test_loss, test_mae, torch.cat(all_preds), torch.cat(all_targets)


# ==================== MAIN ====================
def main():
    # Configurazione
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Dispositivo: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM disponibile: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Parametri (RIDOTTI per velocità)
    window_size = 3  # Ridotto da 5
    hidden_dim = 3   # Ridotto da 5
    n_qubits = 3     # Ridotto da 5
    n_layers = 1     # Ridotto da 2
    batch_size = 512   # Ridotto da 32
    epochs = 5       # Ridotto da 20
    lr = 0.01
    
    # Carica dati
    print("\n📊 Caricamento dataset...")
    data = load_ettm1_data()
    
    # Normalizzazione
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Split
    train_size = int(0.7 * len(data_scaled))
    val_size = int(0.15 * len(data_scaled))
    
    train_data = data_scaled[:train_size]
    val_data = data_scaled[train_size:train_size + val_size]
    test_data = data_scaled[train_size + val_size:]
    
    # Dataset e DataLoader
    train_dataset = TimeSeriesDataset(train_data, window_size)
    val_dataset = TimeSeriesDataset(val_data, window_size)
    test_dataset = TimeSeriesDataset(test_data, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    input_dim = data.shape[1] - 1  # Tutte tranne il target
    output_dim = 1  # Solo il target
    
    # ==================== QGRU ====================
    print("\n" + "="*60)
    print("🌀 TRAINING QUANTUM GRU")
    print("="*60)
    
    qgru_model = QGRU(input_dim, hidden_dim, output_dim, n_qubits, n_layers)
    total_params_qgru = sum(p.numel() for p in qgru_model.parameters())
    quantum_params = sum(p.numel() for name, p in qgru_model.named_parameters() if 'q_params' in name)
    
    print(f"Parametri totali: {total_params_qgru}")
    print(f"Parametri quantistici: {quantum_params}")
    print(f"Parametri classici: {total_params_qgru - quantum_params}")
    
    qgru_train_losses, qgru_val_losses, qgru_times = train_model(
        qgru_model, train_loader, val_loader, epochs, lr, device
    )
    
    qgru_test_loss, qgru_test_mae, qgru_preds, qgru_targets = evaluate_model(
        qgru_model, test_loader, device
    )
    
    print(f"\n✓ QGRU Test MSE: {qgru_test_loss:.6f}")
    print(f"✓ QGRU Test MAE: {qgru_test_mae:.6f}")
    print(f"✓ Tempo medio per epoch: {np.mean(qgru_times):.2f}s")
    
    # ==================== CLASSICAL GRU ====================
    print("\n" + "="*60)
    print("🔄 TRAINING CLASSICAL GRU")
    print("="*60)
    
    classical_gru = ClassicalGRU(input_dim, hidden_dim, output_dim)
    total_params_gru = sum(p.numel() for p in classical_gru.parameters())
    
    print(f"Parametri totali: {total_params_gru}")
    
    gru_train_losses, gru_val_losses, gru_times = train_model(
        classical_gru, train_loader, val_loader, epochs, lr, device
    )
    
    gru_test_loss, gru_test_mae, gru_preds, gru_targets = evaluate_model(
        classical_gru, test_loader, device
    )
    
    print(f"\n✓ GRU Test MSE: {gru_test_loss:.6f}")
    print(f"✓ GRU Test MAE: {gru_test_mae:.6f}")
    print(f"✓ Tempo medio per epoch: {np.mean(gru_times):.2f}s")
    
    # ==================== COMPARISON ====================
    print("\n" + "="*60)
    print("📊 CONFRONTO FINALE")
    print("="*60)
    
    print(f"\n{'Metrica':<25} {'QGRU':<15} {'Classical GRU':<15} {'Differenza'}")
    print("-" * 70)
    print(f"{'Test MSE':<25} {qgru_test_loss:<15.6f} {gru_test_loss:<15.6f} {(qgru_test_loss - gru_test_loss):.6f}")
    print(f"{'Test MAE':<25} {qgru_test_mae:<15.6f} {gru_test_mae:<15.6f} {(qgru_test_mae - gru_test_mae):.6f}")
    print(f"{'Parametri totali':<25} {total_params_qgru:<15} {total_params_gru:<15} {total_params_qgru - total_params_gru}")
    print(f"{'Tempo/epoch (s)':<25} {np.mean(qgru_times):<15.2f} {np.mean(gru_times):<15.2f} {np.mean(qgru_times) - np.mean(gru_times):.2f}")
    
    improvement_mse = ((gru_test_loss - qgru_test_loss) / gru_test_loss) * 100
    print(f"\n{'QGRU improvement in MSE:':<30} {improvement_mse:+.2f}%")
    
    # ==================== VISUALIZATION ====================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training losses
    axes[0, 0].plot(qgru_train_losses, label='QGRU Train', color='blue')
    axes[0, 0].plot(gru_train_losses, label='GRU Train', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation losses
    axes[0, 1].plot(qgru_val_losses, label='QGRU Val', color='blue')
    axes[0, 1].plot(gru_val_losses, label='GRU Val', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Predictions - QGRU
    sample_idx = 0
    axes[1, 0].plot(qgru_targets[sample_idx].flatten(), label='True', color='black', linewidth=2)
    axes[1, 0].plot(qgru_preds[sample_idx].flatten(), label='QGRU Pred', color='blue', linestyle='--')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('QGRU Predictions (Sample)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Predictions - GRU
    axes[1, 1].plot(gru_targets[sample_idx].flatten(), label='True', color='black', linewidth=2)
    axes[1, 1].plot(gru_preds[sample_idx].flatten(), label='GRU Pred', color='red', linestyle='--')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Classical GRU Predictions (Sample)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qgru_vs_gru_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Grafici salvati in 'qgru_vs_gru_comparison.png'")
    
    plt.show()


if __name__ == "__main__":
    main()