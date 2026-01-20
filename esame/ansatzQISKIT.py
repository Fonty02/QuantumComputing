from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import Sampler


options = {'seed': 12345, 'shots': 4096}

def MPS(num_qubits, **kwargs):
    """
    Constructs a Matrix Product State (MPS) quantum circuit.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        **kwargs: Additional keyword arguments to be passed to the
        RealAmplitudes.

    Returns:
        QuantumCircuit: The constructed MPS quantum circuit.

    """
    qc = QuantumCircuit(num_qubits)
    qubits = range(num_qubits)

    # Iterate over adjacent qubit pairs
    for i, j in zip(qubits[:-1], qubits[1:]):
        qc.compose(RealAmplitudes(num_qubits=2,
                                  parameter_prefix=f'θ_{i},{j}',
                                  **kwargs), [i, j],
                   inplace=True)
        qc.barrier(
        )  # Add a barrier after each block for clarity and separation

    return qc

def tensor_ring(num_qubits, **kwargs):
    """
    Constructs a Full Entanglement Tensor Ring quantum circuit.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        **kwargs: Additional keyword arguments to be passed to the
        RealAmplitudes.

    Returns:
        QuantumCircuit: The constructed MPS quantum circuit.

    """
    qc = QuantumCircuit(num_qubits)
    qubits = range(num_qubits)

    # Iterate over adjacent qubit pairs
    for i, j in zip(qubits[:-1], qubits[1:]):
        qc.compose(
            RealAmplitudes(num_qubits=2,
                                  parameter_prefix=f'θ_{i},{j}',
                                  **kwargs), [i, j],
                   inplace=True)
        qc.barrier(
        )  # Add a barrier after each block for clarity and separation

    qc.compose(RealAmplitudes(num_qubits=2,
                              parameter_prefix=f'θ_{num_qubits-1},{0}',
                              **kwargs), [num_qubits-1, 0],
               inplace=True)
    qc.barrier(
    )

    return qc

def _generate_tree_tuples(n):
    """
    Generate a list of tuples representing the tree structure
    of consecutive numbers up to n.

    Args:
        n (int): The number up to which the tuples are generated.

    Returns:
        list: A list of tuples representing the tree structure.
    """
    tuples_list = []
    indices = []

    # Generate initial tuples with consecutive numbers up to n
    for i in range(0, n, 2):
        tuples_list.append((i, i + 1))

    indices += [tuples_list]

    # Perform iterations until we reach a single tuple
    while len(tuples_list) > 1:
        new_tuples = []

        # Generate new tuples by combining adjacent larger numbers
        for i in range(0, len(tuples_list), 2):
            new_tuples.append((tuples_list[i][1], tuples_list[i + 1][1]))

        tuples_list = new_tuples
        indices += [tuples_list]

    return indices


def TTN(num_qubits, **kwargs):
    """
    Constructs a Tree Tensor Network (TTN) quantum circuit.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        **kwargs: Additional keyword arguments to be passed to the
        RealAmplitudes.

    Returns:
        QuantumCircuit: The constructed TTN quantum circuit.

    Raises:
        AssertionError: If the number of qubits is not a power of 2
        or zero.
    """
    qc = QuantumCircuit(num_qubits)

    # Compute qubit indices
    assert num_qubits & (
            num_qubits -
            1) == 0 and num_qubits != 0, "Number of qubits must be a power of 2"

    indices = _generate_tree_tuples(num_qubits)

    # Iterate over each layer of TTN indices
    for layer_indices in indices:
        for i, j in layer_indices:
            qc.compose(RealAmplitudes(num_qubits=2,
                                      parameter_prefix=f'λ_{i},{j}',
                                      **kwargs), [i, j],
                       inplace=True)
        qc.barrier(
        )  # Add a barrier after each layer for clarity and separation

    return qc


def construct_tensor_ring_ansatz_circuit(num_qubits, measured_qubits=0):
    # Function for the construction of the MPS+TTN ansatz
    ansatz = QuantumCircuit(num_qubits, measured_qubits)

    ttn = TTN(num_qubits, reps=1).decompose()
    tr = tensor_ring(num_qubits, reps=1).decompose()

    ansatz.compose(tr, range(num_qubits), inplace=True)
    ansatz.compose(ttn, range(num_qubits), inplace=True)

    if measured_qubits > 0 and measured_qubits == 1:
        # Modify this if needed another type of measurements
        ansatz.measure(num_qubits-1, 0)

    return ansatz


def construct_qnn(feature_map, ansatz, callback_graph=None, maxiter=100, interpret=None, output_shape=2) -> VQC:
    """
    # These lines of code are used to execute circuits on the NVIDIA graphics.
    device = "CUDA" if "GPU" in AerSimulator().available_devices() else "CPU"
    sim = AerSimulator(method='statevector', device=device)
    print('Run on :',AerSimulator().available_devices())
    if device == "CUDA":
        estimator = EstimatorV2(
            options={
                'backend_options': {
                    'method': sim.options.method,
                    'device': sim.options.device,
                    'cuStateVec_enable': True,
                    "seed_simulator": 42
                }
            }
        )
    else:
        estimator = StatevectorEstimator(seed=42)

    # Custom pass-manager
    pm = generate_preset_pass_manager(optimization_level=3, backend=sim)
    """
    sampler = Sampler(options=options)

    # uniform initialization of parameter
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
        #pass_manager=pm
    )

    return classifier