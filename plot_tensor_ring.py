import numpy as np
import pennylane as qml

from esame.tensorRing import tensor_ring_modified


def main():
    num_qubits = 4
    reps = 2
    circuit, num_params = tensor_ring_modified(num_qubits=num_qubits, reps=reps)

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def qnode(params):
        circuit(params)
        return qml.state()

    params = np.zeros(num_params, dtype=float)

    fig, _ = qml.draw_mpl(qnode)(params)
    output_path = "tensor_ring_modified.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
