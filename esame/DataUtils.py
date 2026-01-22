"""
Data Utilities for Parkinsons Telemonitoring Dataset.

This module handles data loading, preprocessing, windowing, and
Leave-One-Group-Out cross-validation for the UCI Parkinsons dataset.
"""

from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


def get_parkinsons_dataset(window_size=10, step=1, normalize="per_fold"):
    """
    Load and preprocess the Parkinsons Telemonitoring dataset (UCI #189).
    
    Creates temporal windows per patient for time series forecasting.
    Target variable: motor_UPDRS
    Excluded features: subject#, age, test_time, sex, total_UPDRS
    
    Args:
        window_size: Number of time steps in each window
        step: Stride between consecutive windows
        normalize: Normalization strategy ('per_fold', 'global', or False)
        
    Returns:
        Tuple of (X_windows, y_windows, groups, logo_splitter, feature_names, scaler)
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    if normalize not in {"per_fold", "global", False, None}:
        raise ValueError("normalize must be 'per_fold', 'global', or False")

    parkinsons_telemonitoring = fetch_ucirepo(id=189)

    X_df = parkinsons_telemonitoring.data.features.copy()
    y_df = parkinsons_telemonitoring.data.targets

    if y_df is None or len(getattr(y_df, "columns", [])) == 0:
        y_df = None
    else:
        y_df = y_df.copy()

    if y_df is None or "motor_UPDRS" not in y_df.columns:
        if "motor_UPDRS" in X_df.columns:
            target_cols = [c for c in ["motor_UPDRS", "total_UPDRS"] if c in X_df.columns]
            y_df = X_df[target_cols].copy()
            X_df = X_df.drop(columns=target_cols)
        else:
            raise ValueError("Column 'motor_UPDRS' not found in data.")

    if "subject#" not in X_df.columns:
        ids_df = getattr(parkinsons_telemonitoring.data, "ids", None)
        original_df = getattr(parkinsons_telemonitoring.data, "original", None)
        if ids_df is not None and "subject#" in ids_df.columns:
            X_df = X_df.join(ids_df[["subject#"]])
        elif original_df is not None and "subject#" in original_df.columns:
            X_df = X_df.join(original_df[["subject#"]])
        else:
            raise ValueError("Column 'subject#' not found in data.")

    if "test_time" not in X_df.columns:
        raise ValueError("Column 'test_time' not found in features.")

    df = X_df.join(y_df[["motor_UPDRS"]])

    drop_features = {"subject#", "age", "test_time", "sex", "total_UPDRS"}
    feature_names = [c for c in X_df.columns if c not in drop_features]

    df = df.dropna(subset=feature_names + ["motor_UPDRS", "subject#", "test_time"])

    X_windows = []
    y_windows = []
    groups = []

    for subject_id in df["subject#"].unique():
        subject_df = df[df["subject#"] == subject_id].sort_values("test_time")
        features = subject_df[feature_names].to_numpy(dtype=float)
        targets = subject_df["motor_UPDRS"].to_numpy(dtype=float)

        n_rows = len(subject_df)
        if n_rows < window_size:
            continue

        for start in range(0, n_rows - window_size + 1, step):
            end = start + window_size
            X_windows.append(features[start:end])
            y_windows.append(targets[end - 1])
            groups.append(subject_id)

    X_windows = np.asarray(X_windows, dtype=float)
    y_windows = np.asarray(y_windows, dtype=float)
    groups = np.asarray(groups)

    scaler = None
    if normalize == "global" and X_windows.size > 0:
        scaler = StandardScaler()
        X_shape = X_windows.shape
        X_flat = X_windows.reshape(-1, X_shape[-1])
        X_flat = scaler.fit_transform(X_flat)
        X_windows = X_flat.reshape(X_shape)

    logo = LeaveOneGroupOut()

    return X_windows, y_windows, groups, logo, feature_names, scaler


def scale_fold(X_windows, train_idx, test_idx):
    """
    Normalize a single fold without data leakage.
    
    Fits scaler only on training data and transforms both train and test.
    
    Args:
        X_windows: Full dataset of windows
        train_idx: Indices for training set
        test_idx: Indices for test set
        
    Returns:
        Tuple of (X_train, X_test, scaler)
    """
    scaler = StandardScaler()
    X_train = X_windows[train_idx]
    X_test = X_windows[test_idx]

    X_shape_train = X_train.shape
    X_shape_test = X_test.shape

    X_train_flat = X_train.reshape(-1, X_shape_train[-1])
    X_test_flat = X_test.reshape(-1, X_shape_test[-1])

    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)

    X_train = X_train_flat.reshape(X_shape_train)
    X_test = X_test_flat.reshape(X_shape_test)

    return X_train, X_test, scaler


def get_parkinsons_logo_folds(window_size=10, step=1):
    """
    Generator for Leave-One-Group-Out cross-validation folds.
    
    Each fold is normalized without data leakage and ready to use.
    
    Args:
        window_size: Number of time steps in each window
        step: Stride between consecutive windows
        
    Yields:
        Dictionary containing:
            - fold_idx: Fold index (0-based)
            - test_patient: Patient ID used for testing
            - X_train, X_test: Normalized feature arrays
            - y_train, y_test: Target arrays
            - feature_names: List of feature names
            - n_train, n_test: Sample counts
            - n_folds: Total number of folds
    """
    X_all, y_all, groups, logo, feature_names, _ = get_parkinsons_dataset(
        window_size=window_size,
        step=step,
        normalize="per_fold"
    )
    
    n_folds = logo.get_n_splits(X_all, y_all, groups)
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups)):
        test_patient = int(groups[test_idx[0]])
        
        X_train, X_test, _ = scale_fold(X_all, train_idx, test_idx)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        
        yield {
            "fold_idx": fold_idx,
            "test_patient": test_patient,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_folds": n_folds,
        }


