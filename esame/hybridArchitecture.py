import torch
import torch.nn as nn
import torch.nn.functional as F
from QLSTM import QLSTM
from QLSTMTensorRing import QLSTMTensorRing

class AttentionLayer(nn.Module):
    """
    Hybrid Architecture using Vaswani Multi-Head Attention.
    This layer applies multi-head self-attention to input sequences and extracts
    a context vector from the last time step.
    """
    def __init__(self, input_dim, num_heads=1):
        super(AttentionLayer, self).__init__()
        
        # Initialize multi-head attention module
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, 
                                         num_heads=num_heads, 
                                         batch_first=True)
        
        # Initialize layer normalization
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        :param x: Input tensor (Batch, Seq_Len, Features)
        :return: context_vector, attention_weights
        """
        # Apply multi-head attention to the input
        attn_output, attn_weights = self.mha(query=x, key=x, value=x)
        
        # Apply residual connection and layer normalization
        x = self.norm(x + attn_output)
        # Extract the context vector from the last time step
        context_vector = x[:, -1, :] 
        
        return context_vector, attn_weights


class HybridQuantumAttentionModel(nn.Module):
    """
    Hybrid Architecture combining Quantum LSTM and Attention Mechanism.
    This model processes time series data by using a quantum LSTM for the main feature
    and incorporating covariates through a multi-head attention layer for prediction.
    """
    def __init__(self, 
                 total_input_features, 
                 hidden_dim, 
                 n_qubits=4, 
                 n_qlayers=1, 
                 model_type='QLSTM',
                 backend='default.qubit',
                 num_heads=1):
        super(HybridQuantumAttentionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        
        # Initialize Quantum LSTM for the main feature (target)
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

        # Calculate number of covariates (additional features)
        self.num_covariates = total_input_features - 1
        
        # Calculate input dimension for attention layer
        self.attention_input_dim = hidden_dim + self.num_covariates
        

        if self.attention_input_dim % num_heads != 0:
            print(f"Warning: attention_input_dim {self.attention_input_dim} not divisible by num_heads {num_heads}. Setting num_heads=1.")
            num_heads = 1
        
        # Initialize attention layer
        self.attention = AttentionLayer(self.attention_input_dim, num_heads=num_heads)
        
        # Initialize regression head for final prediction
        self.regressor = nn.Linear(self.attention_input_dim, 1)

    def forward(self, x):
        """
        :param x: Input tensor (Batch, Seq, Total_Features)
        """
        # 1. Split the input data into target and covariates
        x_target = x[:, :, 0:1]  # Extract target feature (first column)
        
        # 2. Process target feature with Quantum LSTM
        q_out, _ = self.q_lstm(x_target)  # Output: (Batch, Seq, Hidden_Dim)
        
        combined_features = q_out

        # 3. Concatenate covariates directly if present
        if self.num_covariates > 0:
            x_covariates = x[:, :, 1:]  # Extract covariates (remaining columns)
            combined_features = torch.cat((q_out, x_covariates), dim=2)  # Concatenate along feature dimension

        # 4. Apply attention mechanism (Vaswani multi-head attention)
        context_vector, attn_weights = self.attention(combined_features)
        
        # 5. Generate final prediction using regression head
        prediction = self.regressor(context_vector)
        
        return prediction.flatten()  # Return flattened predictions