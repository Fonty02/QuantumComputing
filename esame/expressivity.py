"""
Expressivity Calculation for Variational Quantum Circuits.

This module computes the expressivity of quantum circuits by measuring
the KL-divergence between the circuit's output distribution and the
Haar (random) distribution, following the methodology from:
"Expressibility and Entangling Capability of Parameterized Quantum Circuits"

Lower KL-divergence indicates higher expressivity (circuit covers more of Hilbert space).
"""

import numpy as np
import pennylane as qml
from typing import Callable, Tuple, List
from tqdm import tqdm


def compute_haar_distribution_adaptive(n_qubits: int, n_buckets: int = None) -> Tuple[np.ndarray, List[float]]:
    """
    Compute the Haar distribution.
    
    The Haar distribution for normalized probabilities (p * N where N = 2^n_qubits) 
    
    Bucket boundaries are computed using quantiles of the exponential distribution,
    ensuring each bucket captures an equal portion of the theoretical Haar probability mass.
  
    Args:
        n_qubits: Number of qubits in the circuit
        n_buckets: Number of buckets
        
    Returns:
        Tuple of (haar_probabilities, bucket_boundaries)
    """
    N = 2 ** n_qubits  # Number of basis states
    
    if n_buckets is None:
        n_buckets = N + 1
    
    # Compute bucket boundaries using quantiles of the exponential distribution
    quantiles = np.linspace(0, 1, n_buckets + 1)
    
    boundaries = []
    for q in quantiles:
        if q == 0:
            boundaries.append(0.0)
        elif q >= 1.0 - 1e-10:  # Avoid log(0)
            boundaries.append(np.inf)
        else:
            boundaries.append(-np.log(1 - q))
    
    # Calculate Haar probabilities for each bucket
    haar_probs = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        if np.isinf(b):
            prob = np.exp(-a)
        else:
            prob = np.exp(-a) - np.exp(-b)
        haar_probs.append(prob)
    
    return np.array(haar_probs), boundaries


def assign_to_buckets(normalized_probs: np.ndarray, boundaries: List[float]) -> np.ndarray:
    """
    Assign normalized probabilities to buckets and count occurrences.
    
    Args:
        normalized_probs: Array of normalized probabilities (p * N)
        boundaries: List of bucket boundaries
        
    Returns:
        Array of counts for each bucket
    """
    n_buckets = len(boundaries) - 1
    counts = np.zeros(n_buckets)
    
    for prob in normalized_probs:
        for i in range(n_buckets):
            if boundaries[i] <= prob < boundaries[i + 1]:
                counts[i] += 1
                break
    
    return counts


def kl_divergence(q: np.ndarray, p: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL-divergence D_KL(q || p).
    
    D_KL(q||p) = sum_i q_i * log(q_i / p_i)
    
    Args:
        q: Observed distribution (from circuit)
        p: Reference distribution (Haar)
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL-divergence value
    """
    # Add epsilon to avoid numerical issues
    q = np.clip(q, epsilon, 1.0)
    p = np.clip(p, epsilon, 1.0)
    
    # Normalize to ensure valid probability distributions
    q = q / np.sum(q)
    p = p / np.sum(p)
    
    # Compute KL-divergence
    kl = np.sum(q * np.log(q / p))
    
    return kl


def get_circuit_probabilities_pennylane(
    circuit_fn: Callable,
    params: np.ndarray,
    n_qubits: int,
    wires: list,
    device: qml.device = None
) -> np.ndarray:
    """
    Get output probabilities from a PennyLane circuit.
    
    Args:
        circuit_fn: Function that applies the circuit gates
        params: Circuit parameters
        n_qubits: Number of qubits
        wires: List of wire indices
        device: PennyLane device (if None, creates default.qubit)
        
    Returns:
        Array of probabilities for each basis state
    """
    if device is None:
        device = qml.device("default.qubit", wires=wires)
    
    @qml.qnode(device)
    def prob_circuit(params):
        # Apply the circuit
        circuit_fn(params)
        return qml.probs(wires=wires)
    
    return prob_circuit(params)


def compute_expressivity_single_circuit(
    circuit_fn: Callable,
    n_qubits: int,
    n_params: int,
    n_samples: int = 5000,
    wires: list = None,
    seed: int = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute expressivity for a single quantum circuit.
    
    Follows the methodology from the slides:
    1. Sample random parameters and compute probability distributions
    2. Normalize probabilities by multiplying by N = 2^n_qubits
    3. Assign to buckets and count
    4. Compute Haar distribution for buckets
    5. Calculate KL-divergence
    
    Args:
        circuit_fn: Function that applies the circuit given parameters
        n_qubits: Number of qubits
        n_params: Number of parameters in the circuit
        n_samples: Number of random parameter sets to sample
        wires: List of wire indices (default: range(n_qubits))
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (kl_divergence, observed_distribution, haar_distribution)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if wires is None:
        wires = list(range(n_qubits))
    
    N = 2 ** n_qubits  # Number of basis states
    
    # Get Haar distribution and bucket boundaries
    haar_probs, boundaries = compute_haar_distribution_adaptive(n_qubits)
    n_buckets = len(haar_probs)
    
    # Create device
    device = qml.device("default.qubit", wires=wires)
    
    @qml.qnode(device)
    def prob_circuit(params):
        circuit_fn(params)
        return qml.probs(wires=wires)
    
    # Accumulate bucket counts across all samples
    total_counts = np.zeros(n_buckets)
    
    for _ in tqdm(range(n_samples), desc="Computing expressivity"):
        # Sample random parameters from [0, 2*pi)
        params = np.random.uniform(0, 2 * np.pi, n_params)
        
        # Get probability distribution from circuit
        probs = np.array(prob_circuit(params))
        
        # Normalize: multiply by N
        normalized_probs = probs * N
        
        # Assign to buckets
        counts = assign_to_buckets(normalized_probs, boundaries)
        total_counts += counts
    
    # Convert counts to distribution (normalize by total count)
    total = np.sum(total_counts)
    if total > 0:
        q_distribution = total_counts / total
    else:
        q_distribution = np.ones(n_buckets) / n_buckets
    
    # Compute KL-divergence
    kl = kl_divergence(q_distribution, haar_probs)
    
    return kl, q_distribution, haar_probs


def compute_expressivity_from_model(
    model,
    gate_name: str = 'forget',
    n_samples: int = 5000,
    seed: int = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute expressivity for a QLSTM model using its actual circuit structure.
    
    This function uses the model's get_expressivity_circuit method to ensure
    the exact same circuit structure used during training is evaluated.
    
    Args:
        model: QShallowRegressionLSTM or QShallowRegressionLSTMTensorRing model instance
        gate_name: Name of the gate ('forget', 'input', 'update', 'output')
        n_samples: Number of random parameter sets to sample
        seed: Random seed
        
    Returns:
        Tuple of (kl_divergence, observed_distribution, haar_distribution)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get the LSTM layer from the model
    lstm = model.lstm if hasattr(model, 'lstm') else model
    
    n_qubits = lstm.n_qubits
    N = 2 ** n_qubits
    
    # Get Haar distribution and bucket boundaries (fully adaptive)
    haar_probs, boundaries = compute_haar_distribution_adaptive(n_qubits)
    n_buckets = len(haar_probs)
    
    # Get the probability circuit directly from the model
    # This ensures we use the EXACT same circuit structure as training
    prob_circuit, wires, param_shape = lstm.get_expressivity_circuit(gate_name)
    
    # Accumulate bucket counts
    total_counts = np.zeros(n_buckets)
    
    for _ in tqdm(range(n_samples), desc=f"Computing expressivity for {gate_name} gate"):
        # Random features (simulating input data)
        features = np.random.uniform(-1, 1, n_qubits)
        
        # Random weights
        if isinstance(param_shape, tuple) and len(param_shape) > 1:
            weights = np.random.uniform(0, 2 * np.pi, param_shape)
        else:
            weights = np.random.uniform(0, 2 * np.pi, param_shape[0])
        
        # Get probabilities from the actual model circuit
        probs = np.array(prob_circuit(features, weights))
        
        # Normalize by N (as per methodology from slides)
        normalized_probs = probs * N
        
        # Assign to buckets
        counts = assign_to_buckets(normalized_probs, boundaries)
        total_counts += counts
    
    # Convert to distribution
    total = np.sum(total_counts)
    q_distribution = total_counts / total if total > 0 else np.ones(n_buckets) / n_buckets
    
    # Compute KL-divergence
    kl = kl_divergence(q_distribution, haar_probs)
    
    return kl, q_distribution, haar_probs


def compute_expressivity_for_ansatz(
    ansatz_type: str,
    n_qubits: int,
    n_layers: int = 1,
    n_samples: int = 5000,
    use_modified: bool = False,
    seed: int = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute expressivity for predefined ansatz types.
    
    Args:
        ansatz_type: Type of ansatz ('qlstm', 'tensor_ring', 'tensor_ring_modified')
        n_qubits: Number of qubits
        n_layers: Number of layers/repetitions
        n_samples: Number of samples
        use_modified: For tensor_ring, whether to use modified version
        seed: Random seed
        
    Returns:
        Tuple of (kl_divergence, observed_distribution, haar_distribution)
    """
    if seed is not None:
        np.random.seed(seed)
    
    wires = list(range(n_qubits))
    device = qml.device("default.qubit", wires=wires)
    N = 2 ** n_qubits
    
    # Get Haar distribution
    haar_probs, boundaries = compute_haar_distribution_adaptive(n_qubits)
    n_buckets = len(haar_probs)
    
    if ansatz_type == 'qlstm':
        # Standard QLSTM ansatz
        n_vrotations = 3
        param_shape = (n_layers, n_vrotations, n_qubits)
        
        def ansatz_circuit(params, wires_type):
            for i in range(1, 3):
                for j in range(n_qubits):
                    target = j + i if j + i < n_qubits else j + i - n_qubits
                    qml.CNOT(wires=[wires_type[j], wires_type[target]])
            for i in range(n_qubits):
                qml.RX(params[0][i], wires=wires_type[i])
                qml.RY(params[1][i], wires=wires_type[i])
                qml.RZ(params[2][i], wires=wires_type[i])
        
        @qml.qnode(device)
        def prob_circuit(features, weights):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(features[i], wires=i)
                qml.RZ(features[i] ** 2, wires=i)
            qml.layer(ansatz_circuit, n_layers, weights, wires_type=wires)
            return qml.probs(wires=wires)
            
    elif ansatz_type in ['tensor_ring', 'tensor_ring_modified', 'tensor_ring_rx']:
        from tensorRing import tensor_ring, tensor_ring_modified, tensor_ring_rx
        
        if use_modified or ansatz_type == 'tensor_ring_modified':
            ring_fn, n_params = tensor_ring_modified(n_qubits, reps=n_layers)
        elif ansatz_type == 'tensor_ring_rx':
            ring_fn, n_params = tensor_ring_rx(n_qubits, reps=n_layers)
        else:
            ring_fn, n_params = tensor_ring(n_qubits, reps=n_layers)
        
        param_shape = (n_params,)
        
        @qml.qnode(device)
        def prob_circuit(features, weights):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(features[i], wires=i)
                qml.RZ(features[i] ** 2, wires=i)
            ring_fn(weights)
            return qml.probs(wires=wires)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
    
    # Accumulate bucket counts
    total_counts = np.zeros(n_buckets)
    
    for _ in tqdm(range(n_samples), desc=f"Computing expressivity for {ansatz_type}"):
        features = np.random.uniform(-1, 1, n_qubits)
        
        if isinstance(param_shape, tuple) and len(param_shape) == 3:
            weights = np.random.uniform(0, 2 * np.pi, param_shape)
        else:
            weights = np.random.uniform(0, 2 * np.pi, param_shape[0])
        
        probs = np.array(prob_circuit(features, weights))
        normalized_probs = probs * N
        counts = assign_to_buckets(normalized_probs, boundaries)
        total_counts += counts
    
    total = np.sum(total_counts)
    q_distribution = total_counts / total if total > 0 else np.ones(n_buckets) / n_buckets
    
    kl = kl_divergence(q_distribution, haar_probs)
    
    return kl, q_distribution, haar_probs


def compare_expressivity(
    n_qubits: int = 4,
    n_samples: int = 1000,
    seed: int = 42
) -> dict:
    """
    Compare expressivity across different ansatz configurations.
    
    Args:
        n_qubits: Number of qubits
        n_samples: Number of samples per configuration
        seed: Random seed
        
    Returns:
        Dictionary with expressivity results for each configuration
    """
    results = {}
    
    configurations = [
        ('QLSTM Standard (1 layer)', 'qlstm', 1, False),
        ('QLSTM Standard (2 layers)', 'qlstm', 2, False),
        ('Tensor Ring (1 layer)', 'tensor_ring', 1, False),
        ('Tensor Ring (2 layers)', 'tensor_ring', 2, False),
        ('Tensor Ring Modified (1 layer)', 'tensor_ring', 1, True),
        ('Tensor Ring Modified (2 layers)', 'tensor_ring', 2, True),
        ('Tensor Ring RX (1 layer)', 'tensor_ring_rx', 1, False),
        ('Tensor Ring RX (2 layers)', 'tensor_ring_rx', 2, False),
    ]
    
    for name, ansatz_type, n_layers, use_modified in configurations:
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")
        
        kl, q_dist, haar_dist = compute_expressivity_for_ansatz(
            ansatz_type=ansatz_type,
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_samples=n_samples,
            use_modified=use_modified,
            seed=seed
        )
        
        results[name] = {
            'kl_divergence': kl,
            'observed_distribution': q_dist,
            'haar_distribution': haar_dist,
            'n_qubits': n_qubits,
            'n_layers': n_layers
        }
        
        print(f"KL-divergence: {kl:.6f}")
        print("(Lower is better - more expressive)")
    
    return results


def plot_expressivity_comparison(results: dict, save_path: str = None):
    """
    Plot expressivity comparison results.
    
    Args:
        results: Dictionary from compare_expressivity
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt
    
    names = list(results.keys())
    kl_values = [results[name]['kl_divergence'] for name in names]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of KL-divergences
    ax1 = axes[0]
    bars = ax1.bar(range(len(names)), kl_values, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Ansatz Configuration')
    ax1.set_ylabel('KL-Divergence (lower = more expressive)')
    ax1.set_title('Expressivity Comparison')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars, kl_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Distribution comparison
    ax2 = axes[1]
    width = 0.8 / (len(names) + 1)
    x = np.arange(len(results[names[0]]['haar_distribution']))
    
    # Plot Haar distribution
    ax2.bar(x - 0.4, results[names[0]]['haar_distribution'], width,
            label='Haar (ideal)', color='gold', edgecolor='black')
    
    # Plot each circuit's distribution
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, name in enumerate(names):
        offset = -0.4 + (i + 1) * width
        ax2.bar(x + offset, results[name]['observed_distribution'], width,
                label=name[:20], color=colors[i], edgecolor='black', alpha=0.7)
    
    ax2.set_xlabel('Bucket')
    ax2.set_ylabel('Probability')
    ax2.set_title('Distribution Comparison')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'B{i+1}' for i in range(len(x))])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Expressivity Analysis for Variational Quantum Circuits")
    print("=" * 60)
    
    # Compare different ansatz configurations
    results = compare_expressivity(
        n_qubits=4,
        n_samples=500,  # Use fewer samples for quick demo
        seed=42
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Expressivity Rankings (lower KL = more expressive)")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['kl_divergence'])
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"{rank}. {name}: KL = {data['kl_divergence']:.6f}")
    
    # Plot results
    try:
        plot_expressivity_comparison(results, save_path='esame/plot/expressivity_comparison.png')
    except Exception as e:
        print(f"Could not create plot: {e}")
