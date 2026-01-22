"""
Quantum Ansatz Circuits using Qiskit.

This module provides various tensor network-inspired quantum circuit ansatzes:
- MPS: Matrix Product State
- Tensor Ring: Full entanglement ring topology
- TTN: Tree Tensor Network
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import Sampler


SAMPLER_OPTIONS = {'seed': 12345, 'shots': 4096}


def MPS(num_qubits, **kwargs):
    """
    Construct a Matrix Product State (MPS) quantum circuit.
    
    MPS applies RealAmplitudes blocks to adjacent qubit pairs sequentially.

    Args:
        num_qubits: Number of qubits in the circuit
        **kwargs: Additional arguments for RealAmplitudes
        
    Returns:
        QuantumCircuit with MPS structure
    """
    qc = QuantumCircuit(num_qubits)
    qubits = range(num_qubits)

    for i, j in zip(qubits[:-1], qubits[1:]):
        qc.compose(
            RealAmplitudes(num_qubits=2, parameter_prefix=f'θ_{i},{j}', **kwargs),
            [i, j],
            inplace=True
        )
        qc.barrier()

    return qc


def tensor_ring(num_qubits, **kwargs):
    """
    Construct a Full Entanglement Tensor Ring quantum circuit.
    
    Similar to MPS but with an additional block connecting last and first qubits.

    Args:
        num_qubits: Number of qubits in the circuit
        **kwargs: Additional arguments for RealAmplitudes
        
    Returns:
        QuantumCircuit with Tensor Ring structure
    """
    qc = QuantumCircuit(num_qubits)
    qubits = range(num_qubits)

    for i, j in zip(qubits[:-1], qubits[1:]):
        qc.compose(
            RealAmplitudes(num_qubits=2, parameter_prefix=f'θ_{i},{j}', **kwargs),
            [i, j],
            inplace=True
        )
        qc.barrier()

    qc.compose(
        RealAmplitudes(num_qubits=2, parameter_prefix=f'θ_{num_qubits-1},{0}', **kwargs),
        [num_qubits-1, 0],
        inplace=True
    )
    qc.barrier()

    return qc


def _generate_tree_tuples(n):
    """
    Generate qubit pair indices for Tree Tensor Network structure.
    
    Args:
        n: Number of qubits (must be power of 2)
        
    Returns:
        List of layers, each containing tuples of qubit pairs
    """
    tuples_list = []
    indices = []

    for i in range(0, n, 2):
        tuples_list.append((i, i + 1))

    indices += [tuples_list]

    while len(tuples_list) > 1:
        new_tuples = []
        for i in range(0, len(tuples_list), 2):
            new_tuples.append((tuples_list[i][1], tuples_list[i + 1][1]))
        tuples_list = new_tuples
        indices += [tuples_list]

    return indices


def TTN(num_qubits, **kwargs):
    """
    Construct a Tree Tensor Network (TTN) quantum circuit.
    
    TTN applies RealAmplitudes blocks in a binary tree structure.

    Args:
        num_qubits: Number of qubits (must be power of 2)
        **kwargs: Additional arguments for RealAmplitudes
        
    Returns:
        QuantumCircuit with TTN structure
        
    Raises:
        AssertionError: If num_qubits is not a power of 2
    """
    qc = QuantumCircuit(num_qubits)

    assert num_qubits & (num_qubits - 1) == 0 and num_qubits != 0, \
        "Number of qubits must be a power of 2"

    indices = _generate_tree_tuples(num_qubits)

    for layer_indices in indices:
        for i, j in layer_indices:
            qc.compose(
                RealAmplitudes(num_qubits=2, parameter_prefix=f'λ_{i},{j}', **kwargs),
                [i, j],
                inplace=True
            )
        qc.barrier()

    return qc


def construct_tensor_ring_ansatz_circuit(num_qubits, measured_qubits=0):
    """
    Construct a combined Tensor Ring + TTN ansatz circuit.
    
    Args:
        num_qubits: Number of qubits
        measured_qubits: Number of qubits to measure (0 for none)
        
    Returns:
        QuantumCircuit with combined structure
    """
    ansatz = QuantumCircuit(num_qubits, measured_qubits)

    ttn = TTN(num_qubits, reps=1).decompose()
    tr = tensor_ring(num_qubits, reps=1).decompose()

    ansatz.compose(tr, range(num_qubits), inplace=True)
    ansatz.compose(ttn, range(num_qubits), inplace=True)

    if measured_qubits > 0 and measured_qubits == 1:
        ansatz.measure(num_qubits-1, 0)

    return ansatz


def construct_qnn(feature_map, ansatz, callback_graph=None, maxiter=100, 
                  interpret=None, output_shape=2) -> VQC:
    """
    Construct a Variational Quantum Classifier.
    
    Args:
        feature_map: Quantum circuit for feature encoding
        ansatz: Variational ansatz circuit
        callback_graph: Optional callback for optimization progress
        maxiter: Maximum optimization iterations
        interpret: Output interpretation function
        output_shape: Number of output classes
        
    Returns:
        Configured VQC classifier
    """
    sampler = Sampler(options=SAMPLER_OPTIONS)
    initial_point = [0.5] * ansatz.num_parameters

    classifier = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=maxiter),
        callback=callback_graph,
        interpret=interpret,
        loss='cross_entropy',
        output_shape=output_shape,
        initial_point=initial_point,
    )

    return classifier
