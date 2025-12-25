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
#import scheduler from torch.optim.lr_scheduler
from torch.optim import lr_scheduler

# ==================== 1. CIRCUITO QUANTISTICO FLESSIBILE ====================
class QuantumCircuit:
    def __init__(self, n_qubits, n_layers, ansatz_type='basic', reuploading=False):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz_type = ansatz_type
        self.reuploading = reuploading
        
        # Selezione device (Lightning è consigliato per velocità)
        try:
            self.dev = qml.device('lightning.qubit', wires=n_qubits)
        except:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # Se Re-uploading è False (Paper standard), codifichiamo una volta sola all'inizio
            if not self.reuploading:
                for i in range(self.n_qubits):
                    qml.RX(inputs[i], wires=i)
            
            # Loop sui layer
            for layer in range(self.n_layers):
                # DATA RE-UPLOADING (Solo per My QGRU)
                # Reinseriamo l'input prima di ogni layer variazionale
                if self.reuploading:
                    for i in range(self.n_qubits):
                        qml.RX(inputs[i], wires=i)
                
                # ANSATZ
                if self.ansatz_type == 'basic':
                    # --- Paper Style: Basic Entangler ---
                    # Pesi shape: (n_layers, n_qubits) -> Qui usiamo weights[layer]
                    for i in range(self.n_qubits):
                        qml.RX(weights[layer, i], wires=i)
                    # Entanglement ad anello
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if self.n_qubits > 1:
                        qml.CNOT(wires=[self.n_qubits - 1, 0])
                        
                elif self.ansatz_type == 'strong':
                    # --- My QGRU Style: Strongly Entangling ---
                    # Pesi shape: (n_layers, n_qubits, 3) -> Passiamo la slice del layer
                    # StronglyEntanglingLayers si aspetta (1, n_qubits, 3) per layer singolo
                    # quindi passiamo weights[layer] espanso
                    w_layer = weights[layer].unsqueeze(0) 
                    qml.StronglyEntanglingLayers(w_layer, wires=range(self.n_qubits))
            
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

# ==================== 2. LAYER IBRIDO (CONFIGURABILE) ====================
class VQCLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers, 
                 use_scale_factor=False, ansatz_type='basic', reuploading=False):
        super(VQCLayer, self).__init__()
        self.n_qubits = n_qubits
        self.use_scale_factor = use_scale_factor
        
        self.fc_in = nn.Linear(input_dim, n_qubits)
        
        # Inizializza circuito
        self.qc = QuantumCircuit(n_qubits, n_layers, ansatz_type, reuploading)
        
        # Gestione parametri in base all'Ansatz
        if ansatz_type == 'strong':
            # StronglyEntangling vuole 3 parametri per qubit (Rotazione X, Y, Z)
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        else:
            # BasicEntangler vuole 1 parametro per qubit (Rotazione X)
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
            
        self.fc_out = nn.Linear(n_qubits, hidden_dim)

        # Scale Factor Vettoriale (Uno per qubit) se richiesto
        if self.use_scale_factor:
            self.scale_factor = nn.Parameter(torch.ones(n_qubits))

    def forward(self, x):
        x = self.fc_in(x)
        
        if self.use_scale_factor:
            # My QGRU: Tanh + Scala Vettoriale
            x = torch.tanh(x) * self.scale_factor 
        else:
            # Paper QGRU: Identity (lascia decidere al layer lineare)
            pass 
        
        x = self.qc.forward(x, self.q_params)
        x = self.fc_out(x)
        return x

# ==================== 3. CELLA QGRU ====================
class QGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers, 
                 use_scale_factor, ansatz_type, reuploading):
        super(QGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        concat_dim = hidden_dim + input_dim
        
        # Configurazione condivisa per i 3 gate
        config = {
            'input_dim': concat_dim, 'hidden_dim': hidden_dim,
            'n_qubits': n_qubits, 'n_layers': n_layers,
            'use_scale_factor': use_scale_factor,
            'ansatz_type': ansatz_type, 'reuploading': reuploading
        }
        
        self.vqc_reset = VQCLayer(**config)
        self.vqc_update = VQCLayer(**config)
        self.vqc_output = VQCLayer(**config)
    
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
    def __init__(self, input_dim, hidden_dim, output_dim, n_qubits, n_layers, 
                 use_scale_factor=False, ansatz_type='basic', reuploading=False):
        super(QGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.qgru_cell = QGRUCell(input_dim, hidden_dim, n_qubits, n_layers, 
                                  use_scale_factor, ansatz_type, reuploading)
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
        print("Scaricamento dataset ETTh1...")
        df = pd.read_csv(io.StringIO(requests.get(url, timeout=10).content.decode('utf-8')))]
        return df[['HUFL', 'HULL', 'MUFL', 'OT']].values
    except:
        print("Fallback su dati sintetici...")
        t = np.linspace(0, 100, 2000)
        return np.column_stack([np.sin(t), np.cos(t), np.sin(t)*np.cos(t), np.sin(t+0.5)])

def evaluate_final(model, loader, device):
    model.eval()
    mse_loss, mae_loss = 0, 0
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            mse_loss += criterion_mse(out, y).item()
            mae_loss += criterion_mae(out, y).item()
    return mse_loss / len(loader), mae_loss / len(loader)

def train_experiment(model, train_loader, val_loader, epochs, lr, device, name):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Scheduler come da paper (opzionale se poche epoche, ma corretto averlo)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\n--- Training {name} ---")
    start_total = time.time()
    
    for epoch in range(epochs):
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
        
        scheduler.step()
        avg_train = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                out, _ = model(x.to(device))
                val_loss += criterion(out, y.to(device)).item()
        avg_val = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        print(f"Ep {epoch+1} | Train: {avg_train:.5f} | Val: {avg_val:.5f}")
    
    total_time = time.time()-start_total
    print(f"Total Time {name}: {total_time:.1f}s")
    return history, total_time

# ==================== MAIN ====================
def main():
    device = torch.device("cpu") # CPU consigliata per simulazioni piccoli circuiti
    print(f"Device: {device}")
    
    # --- IPERPARAMETRI (Specifiche Paper) ---
    WINDOW_SIZE = 5
    HIDDEN_DIM = 5  
    N_QUBITS = 5    
    N_LAYERS = 2    
    BATCH_SIZE = 64
    EPOCHS = 1      # Aumenta per risultati migliori (paper usa >100)
    LR = 0.005

    # Dati
    raw = load_data()
    split_idx = int(len(raw) * 0.8)
    
    scaler = MinMaxScaler((-1, 1)).fit(raw[:split_idx])
    data_scaled = scaler.transform(raw)
    
    train_ds = TimeSeriesDataset(data_scaled[:split_idx], WINDOW_SIZE)
    val_ds = TimeSeriesDataset(data_scaled[split_idx:], WINDOW_SIZE)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    dims = (raw.shape[1]-1, HIDDEN_DIM, 1) # (Input, Hidden, Output)
    results_list = []

    # --- 1. MY QGRU (Potenziata: Strong Ent. + Re-upload + Scaling) ---
    print("\n[1/3] Avvio My QGRU (StronglyEntangling + Re-uploading)...")
    my_qgru = QGRU(*dims, N_QUBITS, N_LAYERS, 
                   use_scale_factor=True, 
                   ansatz_type='strong', 
                   reuploading=True)
    hist_my, time_my = train_experiment(my_qgru, train_dl, val_dl, EPOCHS, LR, device, "MY QGRU")
    mse_my, mae_my = evaluate_final(my_qgru, val_dl, device)
    
    results_list.append({
        "Model": "My QGRU (Optimized)",
        "Test MSE": mse_my, "Test MAE": mae_my, "Time (s)": time_my
    })

    # --- 2. PAPER QGRU (Basic Ent. + No Re-upload + No Scaling) ---
    print("\n[2/3] Avvio Paper QGRU (BasicEntangler Standard)...")
    paper_qgru = QGRU(*dims, N_QUBITS, N_LAYERS, 
                      use_scale_factor=False, 
                      ansatz_type='basic', 
                      reuploading=False)
    hist_paper, time_paper = train_experiment(paper_qgru, train_dl, val_dl, EPOCHS, LR, device, "PAPER QGRU")
    mse_paper, mae_paper = evaluate_final(paper_qgru, val_dl, device)
    
    results_list.append({
        "Model": "Paper QGRU",
        "Test MSE": mse_paper, "Test MAE": mae_paper, "Time (s)": time_paper
    })
    
    # --- 3. CLASSICAL GRU ---
    print("\n[3/3] Avvio Classical GRU...")
    classic_gru = ClassicalGRU(*dims)
    hist_classic, time_classic = train_experiment(classic_gru, train_dl, val_dl, EPOCHS, LR, device, "CLASSICAL GRU")
    mse_classic, mae_classic = evaluate_final(classic_gru, val_dl, device)
    
    results_list.append({
        "Model": "Classical GRU",
        "Test MSE": mse_classic, "Test MAE": mae_classic, "Time (s)": time_classic
    })
    
    # --- SALVATAGGIO CSV ---
    df_results = pd.DataFrame(results_list)
    df_results.to_csv("risultati_finali_reupload.csv", index=False)
    print("\n✅ Risultati salvati in 'risultati_finali_reupload.csv'")
    print(df_results)
    
    # --- PLOT ---
    plt.figure(figsize=(12, 5))
    
    # Train Loss
    plt.subplot(1, 2, 1)
    plt.plot(hist_my['train_loss'], label='My QGRU (Strong+ReUp)', marker='o', color='blue')
    plt.plot(hist_paper['train_loss'], label='Paper QGRU (Basic)', marker='s', linestyle='--', color='green')
    plt.plot(hist_classic['train_loss'], label='Classic GRU', marker='^', linestyle=':', color='red')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Val Loss
    plt.subplot(1, 2, 2)
    plt.plot(hist_my['val_loss'], label='My QGRU', marker='o', color='blue')
    plt.plot(hist_paper['val_loss'], label='Paper QGRU', marker='s', linestyle='--', color='green')
    plt.plot(hist_classic['val_loss'], label='Classic GRU', marker='^', linestyle=':', color='red')
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_reupload_strong.png')
    plt.show()

if __name__ == "__main__":
    main()