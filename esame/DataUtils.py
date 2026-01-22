from ucimlrepo import fetch_ucirepo

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


# fetch dataset
def get_parkinsons_dataset(window_size=10, step=1, normalize="per_fold"):
    """
    Carica il Parkinsons Telemonitoring dataset (UCI #189), costruisce
    finestre temporali per paziente e normalizza le feature.

    - Target: motor_UPDRS
    - Feature rimosse: ID (subject#), age, test_time, sex
    - total_UPDRS escluso dai target

    Ritorna:
        X_windows: array (n_samples, window_size, n_features)
        y_windows: array (n_samples,)
        groups: array (n_samples,) con l'ID del paziente per Leave-One-Group-Out
        logo: splitter sklearn LeaveOneGroupOut
        feature_names: lista delle feature usate
        scaler: StandardScaler (solo se normalize="global")
    """
    if window_size <= 0:
        raise ValueError("window_size deve essere > 0")
    if step <= 0:
        raise ValueError("step deve essere > 0")
    if normalize not in {"per_fold", "global", False, None}:
        raise ValueError("normalize deve essere 'per_fold', 'global' o False")

    parkinsons_telemonitoring = fetch_ucirepo(id=189)

    # data (as pandas dataframes)
    X_df = parkinsons_telemonitoring.data.features.copy()
    y_df = parkinsons_telemonitoring.data.targets

    # In alcuni casi i target non sono separati: recuperali dalle feature
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
            raise ValueError("Colonna 'motor_UPDRS' non trovata nei dati.")

    if "subject#" not in X_df.columns:
        ids_df = getattr(parkinsons_telemonitoring.data, "ids", None)
        original_df = getattr(parkinsons_telemonitoring.data, "original", None)
        if ids_df is not None and "subject#" in ids_df.columns:
            X_df = X_df.join(ids_df[["subject#"]])
        elif original_df is not None and "subject#" in original_df.columns:
            X_df = X_df.join(original_df[["subject#"]])
        else:
            raise ValueError("Colonna 'subject#' non trovata nei dati (feature/ids/original).")

    if "test_time" not in X_df.columns:
        raise ValueError("Colonna 'test_time' non trovata nelle feature.")

    # Costruzione tabella con target
    df = X_df.join(y_df[["motor_UPDRS"]])

    # Feature utili (rimozione colonne non utili)
    drop_features = {"subject#", "age", "test_time", "sex", "total_UPDRS"}
    feature_names = [c for c in X_df.columns if c not in drop_features]

    # drop righe con NaN nelle colonne usate
    df = df.dropna(subset=feature_names + ["motor_UPDRS", "subject#", "test_time"])

    # Windowing per paziente
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
            # target = ultimo valore della finestra
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
    Normalizza senza leakage: fit solo sul train di un fold LOGO.
    Ritorna X_train, X_test, scaler.
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
    Carica il dataset Parkinsons e restituisce un generatore di fold 
    per Leave-One-Group-Out cross-validation.
    
    Ogni fold è già normalizzato senza data leakage e pronto all'uso.
    
    Yields:
        Per ogni fold:
        - fold_idx: int, indice del fold (0-based)
        - test_patient: int, ID del paziente usato per il test
        - X_train: array (n_train, window_size, n_features)
        - X_test: array (n_test, window_size, n_features)
        - y_train: array (n_train,)
        - y_test: array (n_test,)
        - feature_names: lista delle feature usate
        - n_train: numero di campioni nel training set
        - n_test: numero di campioni nel test set
    """
    # Carica il dataset completo senza normalizzazione
    X_all, y_all, groups, logo, feature_names, _ = get_parkinsons_dataset(
        window_size=window_size,
        step=step,
        normalize="per_fold"  # placeholder, la normalizzazione sarà per fold
    )
    
    n_folds = logo.get_n_splits(X_all, y_all, groups)
    
    # Itera su ogni fold LOGO
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups)):
        test_patient = int(groups[test_idx[0]])
        
        # Normalizza il fold senza data leakage
        X_train, X_test, scaler = scale_fold(X_all, train_idx, test_idx)
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


    