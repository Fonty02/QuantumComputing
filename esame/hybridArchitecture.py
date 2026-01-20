import torch
import torch.nn as nn
import torch.nn.functional as F
from QLSTM import QLSTM
from QLSTMTensorRing import QLSTMTensorRing

class HybridArchitecture(nn.Module):
    """
    Classical Attention Mechanism.
    Updated to handle dynamic input dimensions (QLSTM state + Raw Covariates).
    """
    def __init__(self, input_dim):
        super(HybridArchitecture, self).__init__()
        # The attention layer maps the combined input (Hidden + Covariates) to a score.
        self.W_q = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        """
        :param x: Combined sequence with shape (Batch, Seq_Len, Hidden_Dim + Covariates)
        :return: context_vector, attention_weights
        """
        # Score calculation: e_t = v * tanh(W * x_t)
        energy = torch.tanh(self.W_q(x)) 
        scores = self.v(energy) # Shape: (Batch, Seq_Len, 1)
        
        # Softmax over the time dimension
        attention_weights = F.softmax(scores, dim=1)
        
        # Context vector is the weighted sum of the inputs
        context_vector = torch.sum(attention_weights * x, dim=1)
        
        return context_vector, attention_weights


class HybridQuantumAttentionModel(nn.Module):
    """
    Hybrid Architecture with Direct Covariate Pass-Through.
    
    Structure:
    1. Main feature (Target) -> Quantum LSTM
    2. Other features (Covariates) -> DIRECTLY to Concatenation
    3. Concatenation (Quantum State + Raw Covariates)
    4. Classical Attention
    5. Regression Head
    """
    def __init__(self, 
                 total_input_features, 
                 hidden_dim, 
                 n_qubits=4, 
                 n_qlayers=1, 
                 model_type='QLSTM',
                 backend='default.qubit'):
        super(HybridQuantumAttentionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        
        # --- Branch A: Quantum LSTM for the Main Feature ---
        if model_type == 'TensorRing':
            self.q_lstm = QLSTMTensorRing(
                input_size=1,
                hidden_size=hidden_dim,
                n_qubits=n_qubits,
                n_qlayers=n_qlayers,
                batch_first=True,
                backend=backend
            )
        else:
            self.q_lstm = QLSTM(
                input_size=1,
                hidden_size=hidden_dim,
                n_qubits=n_qubits,
                n_qlayers=n_qlayers,
                batch_first=True,
                backend=backend
            )

        # Count covariates
        self.num_covariates = total_input_features - 1
        
        # --- Dimensions for Attention ---
        # The input to the attention mechanism is the concatenation of:
        # 1. QLSTM hidden state (hidden_dim)
        # 2. Raw covariate features (num_covariates)
        self.attention_input_dim = hidden_dim + self.num_covariates
        
        self.attention = HybridArchitecture(self.attention_input_dim)
        
        # --- Prediction Head ---
        self.regressor = nn.Linear(self.attention_input_dim, 1)

    def forward(self, x):
        """
        :param x: Input tensor (Batch, Seq, Total_Features)
        """
        # 1. Split the data
        x_target = x[:, :, 0:1] # The univariate target sequence
        
        # 2. Process Target with Quantum LSTM
        q_out, _ = self.q_lstm(x_target) # Shape: (Batch, Seq, Hidden_Dim)
        
        combined_features = q_out

        # 3. Direct Pass of Covariates (Concatenation)
        if self.num_covariates > 0:
            x_covariates = x[:, :, 1:] # Shape: (Batch, Seq, Num_Covariates)
            
            # Concatenate along the feature dimension (dim=2)
            # Result Shape: (Batch, Seq, Hidden_Dim + Num_Covariates)
            combined_features = torch.cat((q_out, x_covariates), dim=2)

        # 4. Apply Attention on the Combined Features
        context_vector, attn_weights = self.attention(combined_features)
        
        # 5. Final Prediction
        prediction = self.regressor(context_vector)
        
        return prediction.flatten()