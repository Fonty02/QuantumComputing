"""
Classic Neural Network Architectures for Time Series Regression.

This module provides classical LSTM-based models for comparison with quantum variants.
"""

import torch
import torch.nn as nn




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


class HybridClassicAttentionModel(nn.Module):
    """
    Hybrid Architecture: Classic LSTM + Multi-Head Attention.
    
    Processes the target feature with LSTM and combines with covariates
    through attention mechanism for final prediction.
    """
    
    def __init__(self, total_input_features, hidden_dim, num_layers=1, num_heads=1):
        super(HybridClassicAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
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
        batch_size = x.shape[0]
        x_target = x[:, :, 0:1]
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        
        lstm_out, _ = self.lstm(x_target, (h0, c0))
        combined_features = lstm_out

        if self.num_covariates > 0:
            x_covariates = x[:, :, 1:]
            combined_features = torch.cat((lstm_out, x_covariates), dim=2)

        context_vector, _ = self.attention(combined_features)
        prediction = self.regressor(context_vector)
        prediction = torch.sigmoid(prediction)
        
        return prediction.flatten()


class ShallowRegressionLSTM(nn.Module):
    """
    Shallow Regression LSTM for direct comparison with quantum variants.
    
    Simple architecture: LSTM -> Linear layer.
    """
    
    def __init__(self, num_sensors, hidden_units, num_layers=1):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Predictions of shape (batch,)
        """
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=x.device)

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[-1]).flatten()
        out = torch.sigmoid(out)
        return out
