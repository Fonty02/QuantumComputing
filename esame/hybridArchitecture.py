"""
Hybrid Quantum-Classical Architecture for Time Series Regression.

This module provides the HybridQuantumAttentionModel that combines
Quantum LSTM with Multi-Head Attention mechanism.
"""

import torch
import torch.nn as nn
from QLSTM import QLSTM
from QLSTMTensorRing import QLSTMTensorRing


class AttentionLayer(nn.Module):
    """Multi-Head Self-Attention Layer based on Vaswani et al."""
    
    def __init__(self, input_dim, num_heads=1):
        super(AttentionLayer, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Forward pass with residual connection and layer normalization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        attn_output, attn_weights = self.mha(query=x, key=x, value=x)
        x = self.norm(x + attn_output)
        context_vector = x[:, -1, :]
        return context_vector, attn_weights


class HybridQuantumAttentionModel(nn.Module):
    """
    Hybrid Architecture: Quantum LSTM + Multi-Head Attention.
    
    Processes the target feature with a Quantum LSTM (standard or Tensor Ring)
    and combines with covariates through attention mechanism for final prediction.
    """
    
    def __init__(self, 
                 total_input_features, 
                 hidden_dim, 
                 n_qubits=4, 
                 n_qlayers=1, 
                 model_type='QLSTM',
                 backend='default.qubit',
                 num_heads=1,
                 ring_variant="standard"):
        super(HybridQuantumAttentionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        
        if model_type == 'TensorRing':
            self.q_lstm = QLSTMTensorRing(
                input_size=1,
                hidden_size=hidden_dim,
                n_qubits=n_qubits,
                n_qlayers=n_qlayers,
                batch_first=True,
                backend=backend,
                ring_variant=ring_variant
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

        self.num_covariates = total_input_features - 1
        self.attention_input_dim = hidden_dim + self.num_covariates

        if self.attention_input_dim % num_heads != 0:
            print(f"Warning: attention_input_dim {self.attention_input_dim} not divisible "
                  f"by num_heads {num_heads}. Setting num_heads=1.")
            num_heads = 1
        
        self.attention = AttentionLayer(self.attention_input_dim, num_heads=num_heads)
        self.regressor = nn.Linear(self.attention_input_dim, 1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, total_features)
            
        Returns:
            Predictions of shape (batch,)
        """
        x_target = x[:, :, 0:1]
        q_out, _ = self.q_lstm(x_target)
        combined_features = q_out

        if self.num_covariates > 0:
            x_covariates = x[:, :, 1:]
            combined_features = torch.cat((q_out, x_covariates), dim=2)

        context_vector, _ = self.attention(combined_features)
        prediction = self.regressor(context_vector)
        prediction = torch.sigmoid(prediction)
        
        return prediction.flatten()
