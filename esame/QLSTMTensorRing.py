"""
Quantum LSTM with Tensor Ring Ansatz.

This module provides a quantum-enhanced LSTM using Tensor Ring circuits
for the variational quantum gates.
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from tensorRing import tensor_ring, tensor_ring_modified


def create_window(df: list, window_size=3):
    """
    Create sliding windows for time series forecasting.
    
    Args:
        df: Input data sequence
        window_size: Size of each window
        
    Returns:
        Tuple of (X, y) arrays for training
    """
    X = []
    y = []

    if len(df) - window_size <= 0:
        return np.asarray(df[:-1]).astype(np.float32), np.asarray(df[-1]).astype(np.float32)

    for i in range(len(df) - window_size):
        row = [a for a in df[i:i + window_size]]
        X.append(row)
        y.append(df[i + window_size])

    return np.asarray(X).astype(np.float32), np.asarray(y).astype(np.float32)


class QLSTMTensorRing(nn.Module):
    """
    Quantum LSTM using Tensor Ring Ansatz.
    
    Replaces classical gate computations with Tensor Ring quantum circuits,
    providing a structured entanglement pattern in a ring topology.
    """
    
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_qubits=4,
                 n_qlayers=1,
                 batch_first=True,
                 return_sequences=False,
                 return_state=False,
                 backend="default.qubit",
                 use_modified_ring=False):
        super(QLSTMTensorRing, self).__init__()
        
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_forget = range(self.n_qubits)
        self.wires_input = range(self.n_qubits)
        self.wires_update = range(self.n_qubits)
        self.wires_output = range(self.n_qubits)

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        if use_modified_ring:
            self.ring_circuit, self.num_ring_params = tensor_ring_modified(
                self.n_qubits, reps=self.n_qlayers
            )
        else:
            self.ring_circuit, self.num_ring_params = tensor_ring(
                self.n_qubits, reps=self.n_qlayers
            )

        def VQC(features, weights, wires_type):
            """Variational Quantum Circuit with feature encoding and tensor ring ansatz."""
            if features.ndim == 1:
                features = features.unsqueeze(0)
            ry_params = [torch.arctan(features[:, i]) for i in range(self.n_qubits)]
            rz_params = [torch.arctan((features[:, i]) ** 2) for i in range(self.n_qubits)]
            
            for i in range(self.n_qubits):
                qml.Hadamard(wires=wires_type[i])
                qml.RY(ry_params[i], wires=wires_type[i])
                qml.RZ(rz_params[i], wires=wires_type[i])

            self.ring_circuit(weights)

        def _circuit_forget(inputs, weights):
            VQC(inputs, weights, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_forget]

        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        def _circuit_input(inputs, weights):
            VQC(inputs, weights, self.wires_input)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_input]

        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        def _circuit_update(inputs, weights):
            VQC(inputs, weights, self.wires_update)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_update]

        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        def _circuit_output(inputs, weights):
            VQC(inputs, weights, self.wires_output)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_output]

        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")

        weight_shapes = {"weights": (self.num_ring_params,)}

        self.clayer_in = torch.nn.Linear(self.concat_size, self.n_qubits)
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, init_states=None):
        """
        Forward pass through the QLSTM with Tensor Ring.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) if batch_first=True
            init_states: Optional tuple of (h_0, c_0) initial states
            
        Returns:
            Tuple of (hidden_sequence, (h_n, c_n))
        """
        if self.batch_first:
            batch_size, seq_length, _ = x.size()
        else:
            seq_length, batch_size, _ = x.size()

        hidden_seq = []
        
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)
            c_t = torch.zeros(batch_size, self.hidden_size)
        else:
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            x_t = x[:, t, :]
            v_t = torch.cat((h_t, x_t), dim=1)
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t)))
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t)))

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

    def get_expressivity_circuit(self, gate_name: str = 'forget'):
        """
        Get a probability-returning circuit for expressivity calculation.
        
        Creates a QNode that replicates the exact Tensor Ring circuit structure
        used in training but returns probabilities instead of expectation values.
        
        Args:
            gate_name: Name of the gate ('forget', 'input', 'update', 'output')
            
        Returns:
            Tuple of (qnode_function, wires, param_shape)
        """
        wires_dict = {
            'forget': self.wires_forget,
            'input': self.wires_input,
            'update': self.wires_update,
            'output': self.wires_output
        }
        wires = list(wires_dict[gate_name])
        
        # Create a fresh device for probability measurement
        device = qml.device("default.qubit", wires=wires)
        
        n_qubits = self.n_qubits
        ring_circuit = self.ring_circuit
        num_ring_params = self.num_ring_params
        
        @qml.qnode(device, interface="numpy")
        def prob_circuit(features, weights):
            """Circuit that returns probabilities for expressivity calculation."""
            # Feature encoding - exact same as VQC in __init__
            ry_params = [np.arctan(features[i]) for i in range(n_qubits)]
            rz_params = [np.arctan(features[i] ** 2) for i in range(n_qubits)]
            
            for i in range(n_qubits):
                qml.Hadamard(wires=wires[i])
                qml.RY(ry_params[i], wires=wires[i])
                qml.RZ(rz_params[i], wires=wires[i])
            
            # Apply the tensor ring circuit - exact same as in training
            ring_circuit(weights)
            
            return qml.probs(wires=wires)
        
        param_shape = (num_ring_params,)
        
        return prob_circuit, wires, param_shape


class QShallowRegressionLSTMTensorRing(nn.Module):
    """
    Shallow Regression model using Quantum LSTM with Tensor Ring.
    
    Simple architecture: QLSTM (Tensor Ring) -> Linear layer for regression tasks.
    """
    
    def __init__(self, num_sensors, hidden_units, n_qubits=4, n_qlayers=1, use_modified_ring=False):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = QLSTMTensorRing(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            use_modified_ring=use_modified_ring
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Predictions of shape (batch,)
        """
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn).flatten()
        out = torch.sigmoid(out)
        return out
