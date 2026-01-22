"""
Tensor Ring Quantum Ansatz Circuits using PennyLane.

This module provides two variants of the Tensor Ring ansatz:
- Standard: Uses only RY rotation gates
- Modified: Uses RY+RZ rotation gates with reduced gates after CNOT
"""

import pennylane as qml


def tensor_ring(num_qubits, reps=1):
    """
    Construct a Full Entanglement Tensor Ring quantum circuit.
    
    The circuit applies parametrized RY rotations on adjacent qubit pairs
    connected by CNOT gates, forming a ring topology.

    Args:
        num_qubits: Number of qubits in the circuit
        reps: Number of repetitions (layers)
        
    Returns:
        Tuple of (circuit_function, num_parameters)
    """
    def circuit(params):
        param_idx = 0
        for _ in range(reps):
            for i in range(num_qubits - 1):
                qml.RY(params[param_idx], wires=i)
                qml.RY(params[param_idx + 1], wires=i + 1)
                param_idx += 2
                qml.CNOT(wires=[i, i + 1])
                qml.RY(params[param_idx], wires=i)
                qml.RY(params[param_idx + 1], wires=i + 1)
                param_idx += 2
                qml.Barrier(wires=range(num_qubits))
            
            # Close the ring: connect last qubit to first
            qml.RY(params[param_idx], wires=num_qubits - 1)
            qml.RY(params[param_idx + 1], wires=0)
            param_idx += 2
            qml.CNOT(wires=[num_qubits - 1, 0])
            qml.RY(params[param_idx], wires=num_qubits - 1)
            qml.RY(params[param_idx + 1], wires=0)
            param_idx += 2
            qml.Barrier(wires=range(num_qubits))
        return param_idx
    
    num_params = 4 * num_qubits * reps
    return circuit, num_params


def tensor_ring_modified(num_qubits, reps=1):
    """
    Construct a Modified Tensor Ring quantum circuit.
    
    Modifications from standard version:
    - RY gates replaced with RY+RZ for increased expressibility
    - Rotational gates removed after CNOT (reduced parameter count per block)

    Args:
        num_qubits: Number of qubits in the circuit
        reps: Number of repetitions (layers)
        
    Returns:
        Tuple of (circuit_function, num_parameters)
    """
    def circuit(params):
        param_idx = 0
        for _ in range(reps):
            for i in range(num_qubits - 1):
                qml.RY(params[param_idx], wires=i)
                qml.RZ(params[param_idx + 1], wires=i)
                qml.RY(params[param_idx + 2], wires=i + 1)
                qml.RZ(params[param_idx + 3], wires=i + 1)
                param_idx += 4
                qml.CNOT(wires=[i, i + 1])
                param_idx += 2
                qml.Barrier(wires=range(num_qubits))
            
            # Close the ring: connect last qubit to first
            qml.RY(params[param_idx], wires=num_qubits - 1)
            qml.RZ(params[param_idx + 1], wires=num_qubits - 1)
            qml.RY(params[param_idx + 2], wires=0)
            qml.RZ(params[param_idx + 3], wires=0)
            param_idx += 4
            qml.CNOT(wires=[num_qubits - 1, 0])
            param_idx += 2
            qml.Barrier(wires=range(num_qubits))
        return param_idx
    
    num_params = 6 * num_qubits * reps
    return circuit, num_params
