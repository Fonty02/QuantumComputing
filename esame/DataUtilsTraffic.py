import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def get_traffic_dataset(window_size=4, train_samples=-1):
    """
    Download and process a traffic dataset (Metro Interstate Traffic Volume from UCI).
    Target: Traffic volume (goes to QLSTM).
    Covariates: Temperature, Rain, Clouds, Hour of the day (go to Attention mechanism).
    
    Args:
        window_size (int): Size of the sliding window for time series.
        train_samples (int): Number of samples to use for training. 
                             If -1, use the entire dataset.
    
    Returns:
        X_tensor: Input features tensor of shape (num_samples, window_size, num_features)
        y_tensor: Target values tensor of shape (num_samples,)
        scaler: MinMaxScaler object used for normalization
    """
    
    # 1. Dataset Download (Metro Interstate Traffic Volume - UCI)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
    print(f"Downloading dataset from: {url}")
    
    try:
        df = pd.read_csv(url, compression='gzip')
    except Exception as e:
        print("Error in direct download. Make sure you have internet connection.")
        return None, None, None

    # 2. Preprocessing and Feature Engineering
    # Convert the date to extract the hour (fundamental covariate for traffic)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    
    # Sort by date
    df = df.sort_values('date_time').reset_index(drop=True)

    # Select the columns
    # Target: 'traffic_volume' (Goes to QLSTM)
    # Covariates: 'temp', 'rain_1h', 'clouds_all', 'hour' (Go to Attention)
    
    # Note: We remove holidays or weather_main strings for numerical simplicity
    features_numeric = ['traffic_volume', 'temp', 'rain_1h', 'clouds_all', 'hour']
    df_subset = df[features_numeric].copy()
    
    # Handle anomalous values (e.g., temperature at 0 Kelvin)
    df_subset = df_subset[df_subset['temp'] > 200] 
    
    # 3. Slice for "Short Time Series" (Quantum Simulation)
    # Take only a consecutive subset (e.g., the first 300 hours)
    # This drastically reduces the training time of the quantum circuit.
    if train_samples!= -1:
        df_short = df_subset.iloc[:train_samples + window_size + 10]
    
    print(f"Dataset reduced to {len(df_short)} samples for quantum simulation.")

    # 4. Normalization (MinMax scaling between 0 and 1)
    # Essential because QLSTM often uses limited activations (0,1) or (-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df_short)
    
    # data_scaled column 0 is 'traffic_volume' (Target)
    # data_scaled columns 1,2,3,4 are the covariates
    
    # 5. Create sliding window sequences
    X, y = [], []
    for i in range(len(data_scaled) - window_size):
        # Input: (window_size, all features)
        row = data_scaled[i : i + window_size]
        X.append(row)
        
        # Output: Only the target (traffic_volume) at the next time step
        label = data_scaled[i + window_size, 0] 
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor, scaler

