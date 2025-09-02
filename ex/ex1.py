# Installazione (da terminale):
# pip install qiskit qiskit-aer

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Funzione per eseguire il circuito e salvare i risultati
def execute_and_plot(qc, filename):
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    job = simulator.run(compiled, shots=1024)
    result = job.result()
    counts = result.get_counts(qc)
    print(f"Risultati per {filename}: {counts}")  # Stampa i risultati
    qc.draw(output='mpl', filename=filename)
    plot_histogram(counts).savefig(filename.replace('.png', '_results.png'))

# Circuito base
q = QuantumRegister(2)
c = ClassicalRegister(2)
qc = QuantumCircuit(q, c)
qc.measure(q, c)
execute_and_plot(qc, 'circuit.png')

# Hadamard Gate
qc = QuantumCircuit(q, c)
qc.h(q[0])
qc.measure(q, c)
execute_and_plot(qc, 'circuit_h.png')

# X Gate
qc = QuantumCircuit(q, c)
qc.x(q[0])
qc.measure(q, c)
execute_and_plot(qc, 'circuit_x.png')

# Y Gate
qc = QuantumCircuit(q, c)
qc.y(q[0])
qc.measure(q, c)
execute_and_plot(qc, 'circuit_y.png')

# Z Gate
qc = QuantumCircuit(q, c)
qc.z(q[0])
qc.measure(q, c)
execute_and_plot(qc, 'circuit_z.png')

# CNOT Gate
qc = QuantumCircuit(q, c)
qc.cx(q[0], q[1])
qc.measure(q, c)
execute_and_plot(qc, 'circuit_cnot.png')

#combinatio
qc = QuantumCircuit(q, c)
qc.x(q[0])
qc.z(q[0])
qc.measure(q, c)
execute_and_plot(qc, 'circuit_xz.png')
