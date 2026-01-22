import pennylane as qml
# This module defines the Tensor Ring quantum ansatz circuit using PennyLane,



def tensor_ring(num_qubits, reps=1):
    """
    Constructs a Full Entanglement Tensor Ring quantum circuit.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        **kwargs: Additional keyword arguments to be passed to the
        RealAmplitudes.
    Returns:
        a circuit function and the number of parameters.
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
            
            # Closing the ring
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
    Constructs a Modified Tensor Ring quantum circuit.
    Modifications: RY gates replaced with RY+RZ, last rotational gate removed after CNOT.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        **kwargs: Additional keyword arguments to be passed to the
        RealAmplitudes.
    Returns:
        a circuit function and the number of parameters.
    """
    def circuit(params):
        param_idx = 0
        for _ in range(reps):
            for i in range(num_qubits - 1):
                # Apply RY + RZ on both qubits before CNOT
                qml.RY(params[param_idx], wires=i)
                qml.RZ(params[param_idx + 1], wires=i)
                qml.RY(params[param_idx + 2], wires=i + 1)
                qml.RZ(params[param_idx + 3], wires=i + 1)
                param_idx += 4
                qml.CNOT(wires=[i, i + 1])
                # Apply RY + RZ only on first qubit after CNOT (removing last gate)
                param_idx += 2
                qml.Barrier(wires=range(num_qubits))
            
            # Closing the ring
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