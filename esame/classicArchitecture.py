import torch
import torch.nn as nn


class ClassicLSTM(nn.Module):
    """
    Classic LSTM model for regression tasks.
    This model uses a standard LSTM layer followed by a linear layer for prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(ClassicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Standard LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        
        # Linear layer for regression output
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        :param x: Input tensor (Batch, Seq_Len, Features)
        :return: Predictions (Batch,)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Pass through LSTM
        _, (hn, _) = self.lstm(x, (h0, c0))
        
        # Use the last layer's hidden state for prediction
        out = self.linear(hn[-1]).flatten()
        return out


class AttentionLayer(nn.Module):
    """
    Multi-Head Attention Layer using Vaswani's architecture.
    """
    def __init__(self, input_dim, num_heads=1):
        super(AttentionLayer, self).__init__()
        
        # Initialize multi-head attention module
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Initialize layer normalization
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        :param x: Input tensor (Batch, Seq_Len, Features)
        :return: context_vector, attention_weights
        """
        # Apply multi-head attention
        attn_output, attn_weights = self.mha(query=x, key=x, value=x)
        
        # Apply residual connection and layer normalization
        x = self.norm(x + attn_output)
        
        # Extract context vector from the last time step
        context_vector = x[:, -1, :]
        
        return context_vector, attn_weights


class HybridClassicAttentionModel(nn.Module):
    """
    Hybrid Architecture combining Classic LSTM and Attention Mechanism.
    This model processes time series data by using a classic LSTM for the main feature
    and incorporating covariates through a multi-head attention layer for prediction.
    """
    def __init__(self, 
                 total_input_features, 
                 hidden_dim, 
                 num_layers=1,
                 num_heads=1):
        super(HybridClassicAttentionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initialize Classic LSTM for the main feature (target)
        self.lstm = nn.LSTM(
            input_size=1,  # Only target feature
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
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
        batch_size = x.shape[0]
        
        # 1. Split the input data into target and covariates
        x_target = x[:, :, 0:1]  # Extract target feature (first column)
        
        # 2. Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        
        # 3. Process target feature with Classic LSTM
        lstm_out, _ = self.lstm(x_target, (h0, c0))  # Output: (Batch, Seq, Hidden_Dim)
        
        combined_features = lstm_out

        # 4. Concatenate covariates directly if present
        if self.num_covariates > 0:
            x_covariates = x[:, :, 1:]  # Extract covariates (remaining columns)
            combined_features = torch.cat((lstm_out, x_covariates), dim=2)

        # 5. Apply attention mechanism
        context_vector, attn_weights = self.attention(combined_features)
        
        # 6. Generate final prediction using regression head
        prediction = self.regressor(context_vector)
        
        return prediction.flatten()


class ShallowRegressionLSTM(nn.Module):
    """
    Shallow regression model using Classic LSTM.
    Simple architecture: LSTM -> Linear for direct comparison with quantum variants.
    """
    def __init__(self, num_sensors, hidden_units, num_layers=1):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        # Initialize the LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True
        )

        # Linear layer for regression output
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=x.device)

        # Pass through LSTM and get final hidden state
        _, (hn, _) = self.lstm(x, (h0, c0))
        
        # Apply linear layer and flatten for output
        out = self.linear(hn[-1]).flatten()
        return out
