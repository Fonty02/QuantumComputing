import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import requests
import io
import time
from tqdm import tqdm

# ==================== 1. CIRCUITO QUANTISTICO ====================
class QuantumCircuit:
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Forza CPU per velocità su piccoli circuiti
        try:
            self.dev = qml.device('lightning.qubit', wires=n_qubits)
        except:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # Angle Encoding
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            
            # Ansatz: Basic Entangler Layers (Fig. 4b del paper)
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i], wires=i)
                # Entanglement ad anello
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
            res = self.circuit(inputs[i], weights)
            if isinstance(res, list):
                res = torch.stack(res)
            results.append(res)
        return torch.stack(results).float()

# ==================== 2. LAYER IBRIDO (MODIFICATO PER CONFRONTO) ====================
class VQCLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers, use_scale_factor=True):
        super(VQCLayer, self).__init__()
        self.n_qubits = n_qubits
        self.use_scale_factor = use_scale_factor
        
        self.fc_in = nn.Linear(input_dim, n_qubits)
        self.qc = QuantumCircuit(n_qubits, n_layers)
        self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.fc_out = nn.Linear(n_qubits, hidden_dim)

        # IL TUO TRUCCO: Parametro di scala apprendibile
        if self.use_scale_factor:
            self.scale_factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.fc_in(x)
        
        if self.use_scale_factor:
            # APPROCCIO "MY QGRU": Tanh controllata + Scala
            x = torch.tanh(x) * self.scale_factor 
        else:
            # APPROCCIO "PAPER QGRU": Nessuna attivazione forzata.
            # Il layer lineare impara direttamente gli angoli θ.
            # Poiché Rx è periodica, il network può imparare qualsiasi rotazione.
            pass 
        
        x = self.qc.forward(x, self.q_params)
        x = self.fc_out(x)
        return x

# ==================== 3. CELLA QGRU ====================
class QGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers, use_scale_factor):
        super(QGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        concat_dim = hidden_dim + input_dim
        
        # Passiamo il flag use_scale_factor ai layer
        self.vqc_reset = VQCLayer(concat_dim, hidden_dim, n_qubits, n_layers, use_scale_factor)
        self.vqc_update = VQCLayer(concat_dim, hidden_dim, n_qubits, n_layers, use_scale_factor)
        self.vqc_output = VQCLayer(concat_dim, hidden_dim, n_qubits, n_layers, use_scale_factor)
    
    def forward(self, x, h_prev):
        combined = torch.cat([h_prev, x], dim=1)
        
        r_t = torch.sigmoid(self.vqc_reset(combined))
        z_t = torch.sigmoid(self.vqc_update(combined))
        
        combined_reset = torch.cat([r_t * h_prev, x], dim=1)
        h_tilde = torch.tanh(self.vqc_output(combined_reset))
        
        h = z_t * h_prev + (1 - z_t) * h_tilde
        return h

# ==================== 4. MODELLO QGRU COMPLETO ====================
class QGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_qubits, n_layers, use_scale_factor=True):
        super(QGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.qgru_cell = QGRUCell(input_dim, hidden_dim, n_qubits, n_layers, use_scale_factor)
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

# ==================== 5. CLASSICAL GRU ====================
class ClassicalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassicalGRU, self).__init__()
        # Batch_first=True per coerenza
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, h = self.gru(x)
        out = self.fc_out(out)
        return out, h

# ==================== UTILS & DATA ====================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
    def __len__(self): return len(self.data) - self.window_size
    def __getitem__(self, idx):
        return self.data[idx:idx+self.window_size, :-1], self.data[idx+1:idx+self.window_size+1, -1:]

def load_data():
    try:
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
        df = pd.read_csv(io.StringIO(requests.get(url, timeout=5).content.decode('utf-8')))
        return df[['HUFL', 'HULL', 'MUFL', 'OT']].values
    except:
        print("Uso dati sintetici...")
        t = np.linspace(0, 100, 2000)
        return np.column_stack([np.sin(t), np.cos(t), np.sin(t)*np.cos(t), np.sin(t+0.5)])
import pandas as pd # Assicurati di avere pandas installato

# ... [MANTIENI LE CLASSI QuantumCircuit, VQCLayer, QGRU, ClassicalGRU, Dataset INVARIATE] ...

# ==================== FUNZIONE DI VALUTAZIONE FINALE ====================
def evaluate_final(model, loader, device):
    """Calcola MSE e MAE finali sul set di validazione/test"""
    model.eval()
    mse_loss = 0
    mae_loss = 0
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            mse_loss += criterion_mse(out, y).item()
            mae_loss += criterion_mae(out, y).item()
            
    avg_mse = mse_loss / len(loader)
    avg_mae = mae_loss / len(loader)
    return avg_mse, avg_mae

# ==================== TRAINING LOOP (AGGIORNATO) ====================
def train_experiment(model, train_loader, val_loader, epochs, lr, device, name):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\n--- Training {name} ---")
    start_total = time.time()
    
    for epoch in range(epochs):
        ep_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                out, _ = model(x.to(device))
                val_loss += criterion(out, y.to(device)).item()
        avg_val = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        print(f"Ep {epoch+1} | Train: {avg_train:.5f} | Val: {avg_val:.5f} | Time: {time.time()-ep_start:.1f}s")
    
    total_time = time.time()-start_total
    print(f"Total Time {name}: {total_time:.1f}s")
    return history, total_time

# ==================== MAIN AGGIORNATO ====================
def main():
    # Setup Device
    device = torch.device("cpu") 
    print(f"Device: {device}")
    
    # Parametri (Paper: Q=5, L=2, H=5)
    WINDOW_SIZE = 5
    HIDDEN_DIM = 5  
    N_QUBITS = 5    
    N_LAYERS = 2    
    BATCH_SIZE = 128
    EPOCHS = 15       # Metti almeno 3 epoche per vedere una linea nel grafico
    LR = 0.005

    # Dati
    raw = load_data()
    split_idx = int(len(raw) * 0.8)
    
    scaler = MinMaxScaler((-1, 1)).fit(raw[:split_idx])
    data_scaled = scaler.transform(raw)
    
    train_ds = TimeSeriesDataset(data_scaled[:split_idx], WINDOW_SIZE)
    val_ds = TimeSeriesDataset(data_scaled[split_idx:], WINDOW_SIZE) # Usiamo Val come Test per semplicità
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    dims = (raw.shape[1]-1, HIDDEN_DIM, 1)
    
    results_list = []

    # --- 1. MY QGRU ---
    my_qgru = QGRU(*dims, N_QUBITS, N_LAYERS, use_scale_factor=True)
    hist_my, time_my = train_experiment(my_qgru, train_dl, val_dl, EPOCHS, LR, device, "MY QGRU (Scaled)")
    mse_my, mae_my = evaluate_final(my_qgru, val_dl, device)
    
    results_list.append({
        "Model": "My QGRU (Scaled)",
        "Test MSE": mse_my,
        "Test MAE": mae_my,
        "Training Time (s)": time_my,
        "Scale Reset": my_qgru.qgru_cell.vqc_reset.scale_factor.item(),
        "Scale Update": my_qgru.qgru_cell.vqc_update.scale_factor.item(),
        "Scale Output": my_qgru.qgru_cell.vqc_output.scale_factor.item()
    })
    
    # --- 2. PAPER QGRU ---
    paper_qgru = QGRU(*dims, N_QUBITS, N_LAYERS, use_scale_factor=False)
    hist_paper, time_paper = train_experiment(paper_qgru, train_dl, val_dl, EPOCHS, LR, device, "PAPER QGRU")
    mse_paper, mae_paper = evaluate_final(paper_qgru, val_dl, device)
    
    results_list.append({
        "Model": "Paper QGRU",
        "Test MSE": mse_paper,
        "Test MAE": mae_paper,
        "Training Time (s)": time_paper,
        "Scale Reset": "N/A", "Scale Update": "N/A", "Scale Output": "N/A"
    })
    
    # --- 3. CLASSICAL GRU ---
    classic_gru = ClassicalGRU(*dims)
    hist_classic, time_classic = train_experiment(classic_gru, train_dl, val_dl, EPOCHS, LR, device, "CLASSICAL GRU")
    mse_classic, mae_classic = evaluate_final(classic_gru, val_dl, device)
    
    results_list.append({
        "Model": "Classical GRU",
        "Test MSE": mse_classic,
        "Test MAE": mae_classic,
        "Training Time (s)": time_classic,
        "Scale Reset": "N/A", "Scale Update": "N/A", "Scale Output": "N/A"
    })
    
    # --- SALVATAGGIO CSV ---
    df_results = pd.DataFrame(results_list)
    df_results.to_csv("risultati_finali.csv", index=False)
    print("\n✅ Risultati salvati in 'risultati_finali.csv'")
    print(df_results[['Model', 'Test MSE', 'Test MAE']])
    
    # --- PLOT FIXATO (Con markers) ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Aggiunto marker='o' per vedere i punti anche se le epoche sono poche
    plt.plot(hist_my['train_loss'], label='My QGRU', marker='o') 
    plt.plot(hist_paper['train_loss'], label='Paper QGRU', marker='s', linestyle='--')
    plt.plot(hist_classic['train_loss'], label='Classical', marker='^', linestyle=':')
    plt.title(f"Train Loss (Epochs={EPOCHS})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(hist_my['val_loss'], label='My QGRU', marker='o')
    plt.plot(hist_paper['val_loss'], label='Paper QGRU', marker='s', linestyle='--')
    plt.plot(hist_classic['val_loss'], label='Classical', marker='^', linestyle=':')
    plt.title("Val Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png')

if __name__ == "__main__":
    main()