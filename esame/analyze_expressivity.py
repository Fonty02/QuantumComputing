"""
Standalone script for Expressivity Analysis.

This script computes and compares the expressivity of different quantum circuit ansatzes
using the KL-divergence methodology from the course slides.

Usage:
    python analyze_expressivity.py [--n_qubits N] [--n_samples N] [--seed S]
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expressivity import (
    compute_expressivity_for_ansatz,
    compare_expressivity,
    plot_expressivity_comparison
)


def analyze_expressivity_by_qubits(
    qubit_range: list = [2, 3, 4, 5, 6],
    n_samples: int = 500,
    seed: int = 42
):
    """
    Analyze how expressivity changes with the number of qubits.
    
    Args:
        qubit_range: List of qubit counts to analyze
        n_samples: Number of samples per configuration
        seed: Random seed
    """
    import matplotlib.pyplot as plt
    
    results_by_qubits = {}
    
    ansatzes = [
        ('QLSTM', 'qlstm', 1, False),
        ('Tensor Ring', 'tensor_ring', 1, False),
        ('Tensor Ring Modified', 'tensor_ring', 1, True),
    ]
    
    for n_qubits in qubit_range:
        print(f"\n{'='*60}")
        print(f"Analyzing with {n_qubits} qubits")
        print(f"{'='*60}")
        
        results_by_qubits[n_qubits] = {}
        
        for name, ansatz_type, n_layers, use_modified in ansatzes:
            print(f"\n  Computing {name}...")
            try:
                kl, q_dist, haar_dist = compute_expressivity_for_ansatz(
                    ansatz_type=ansatz_type,
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    n_samples=n_samples,
                    use_modified=use_modified,
                    seed=seed
                )
                results_by_qubits[n_qubits][name] = kl
                print(f"    KL-divergence: {kl:.6f}")
            except Exception as e:
                print(f"    Error: {e}")
                results_by_qubits[n_qubits][name] = None
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, _, _, _ in ansatzes:
        kl_values = []
        valid_qubits = []
        for n_qubits in qubit_range:
            kl = results_by_qubits[n_qubits].get(name)
            if kl is not None:
                kl_values.append(kl)
                valid_qubits.append(n_qubits)
        
        if kl_values:
            ax.plot(valid_qubits, kl_values, marker='o', linewidth=2, label=name)
    
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('KL-Divergence (lower = more expressive)', fontsize=12)
    ax.set_title('Expressivity vs Number of Qubits', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'plot', 'expressivity_by_qubits.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    plt.show()
    
    return results_by_qubits


def analyze_expressivity_by_layers(
    n_qubits: int = 4,
    layer_range: list = [1, 2, 3, 4],
    n_samples: int = 500,
    seed: int = 42
):
    """
    Analyze how expressivity changes with the number of layers.
    
    Args:
        n_qubits: Number of qubits
        layer_range: List of layer counts to analyze
        n_samples: Number of samples per configuration
        seed: Random seed
    """
    import matplotlib.pyplot as plt
    
    results_by_layers = {}
    
    ansatzes = [
        ('QLSTM', 'qlstm', False),
        ('Tensor Ring', 'tensor_ring', False),
        ('Tensor Ring Modified', 'tensor_ring', True),
    ]
    
    for n_layers in layer_range:
        print(f"\n{'='*60}")
        print(f"Analyzing with {n_layers} layer(s)")
        print(f"{'='*60}")
        
        results_by_layers[n_layers] = {}
        
        for name, ansatz_type, use_modified in ansatzes:
            print(f"\n  Computing {name}...")
            try:
                kl, q_dist, haar_dist = compute_expressivity_for_ansatz(
                    ansatz_type=ansatz_type,
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    n_samples=n_samples,
                    use_modified=use_modified,
                    seed=seed
                )
                results_by_layers[n_layers][name] = kl
                print(f"    KL-divergence: {kl:.6f}")
            except Exception as e:
                print(f"    Error: {e}")
                results_by_layers[n_layers][name] = None
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, _, _ in ansatzes:
        kl_values = []
        valid_layers = []
        for n_layers in layer_range:
            kl = results_by_layers[n_layers].get(name)
            if kl is not None:
                kl_values.append(kl)
                valid_layers.append(n_layers)
        
        if kl_values:
            ax.plot(valid_layers, kl_values, marker='o', linewidth=2, label=name)
    
    ax.set_xlabel('Number of Layers', fontsize=12)
    ax.set_ylabel('KL-Divergence (lower = more expressive)', fontsize=12)
    ax.set_title(f'Expressivity vs Number of Layers (n_qubits={n_qubits})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'plot', 'expressivity_by_layers.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    plt.show()
    
    return results_by_layers


def main():
    parser = argparse.ArgumentParser(description='Analyze expressivity of quantum circuits')
    parser.add_argument('--n_qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, default='compare', 
                       choices=['compare', 'by_qubits', 'by_layers', 'all'],
                       help='Analysis mode')
    
    args = parser.parse_args()
    
    print("="*70)
    print("EXPRESSIVITY ANALYSIS FOR VARIATIONAL QUANTUM CIRCUITS")
    print("="*70)
    print("\nMethodology: KL-divergence w.r.t. Haar distribution")
    print("Lower KL-divergence = Higher expressivity = Better coverage of Hilbert space")
    print("\nFrom the slides:")
    print("  - Low expressivity → circuit cannot represent target function (underfitting)")
    print("  - High expressivity → circuit too flexible (training unstable, poor generalization)")
    print("="*70)
    
    if args.mode in ['compare', 'all']:
        print("\n" + "="*70)
        print("COMPARISON OF DIFFERENT ANSATZ CONFIGURATIONS")
        print("="*70)
        
        results = compare_expressivity(
            n_qubits=args.n_qubits,
            n_samples=args.n_samples,
            seed=args.seed
        )
        
        # Print ranking
        print("\n" + "="*70)
        print("EXPRESSIVITY RANKING (lower KL = more expressive)")
        print("="*70)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['kl_divergence'])
        for rank, (name, data) in enumerate(sorted_results, 1):
            print(f"  {rank}. {name}: KL = {data['kl_divergence']:.6f}")
        
        try:
            plot_path = os.path.join(os.path.dirname(__file__), 'plot', 'expressivity_comparison.png')
            plot_expressivity_comparison(results, save_path=plot_path)
        except Exception as e:
            print(f"Could not create plot: {e}")
    
    if args.mode in ['by_qubits', 'all']:
        print("\n" + "="*70)
        print("EXPRESSIVITY VS NUMBER OF QUBITS")
        print("="*70)
        
        analyze_expressivity_by_qubits(
            qubit_range=[2, 3, 4, 5],
            n_samples=args.n_samples,
            seed=args.seed
        )
    
    if args.mode in ['by_layers', 'all']:
        print("\n" + "="*70)
        print("EXPRESSIVITY VS NUMBER OF LAYERS")
        print("="*70)
        
        analyze_expressivity_by_layers(
            n_qubits=args.n_qubits,
            layer_range=[1, 2, 3],
            n_samples=args.n_samples,
            seed=args.seed
        )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
