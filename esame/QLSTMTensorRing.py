# Import necessary libraries
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from tensorRing import tensor_ring

def create_window(df: list, window_size=3):
    """
    function used to create the window for forecasting task
    :param df: matrix containing all the data for all the weight
    :param window_size: window size for forecasting
    :return: df: dataframe containing all data for forecasting
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


# Define the QLSTMTensorRing class, a quantum LSTM model using Tensor Ring circuits
class QLSTMTensorRing(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_qubits=4,
                 n_qlayers=1,
                 batch_first=True,
                 return_sequences=False,
                 return_state=False,
                 backend="default.qubit"):
        super(QLSTMTensorRing, self).__init__()
        # Set model parameters
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend

        # Set flags for batch processing and output
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Define wires for different gates in the quantum circuit
        self.wires_forget = range(self.n_qubits)
        self.wires_input = range(self.n_qubits)
        self.wires_update = range(self.n_qubits)
        self.wires_output = range(self.n_qubits)

        # Create quantum devices for each gate
        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        # Get the tensor ring circuit and number of parameters
        self.ring_circuit, self.num_ring_params = tensor_ring(self.n_qubits, reps=self.n_qlayers)

        # Define the Variational Quantum Circuit (VQC) function
        def VQC(features, weights, wires_type):
            # Encode features into rotation parameters
            ry_params = [torch.arctan(feature) for feature in features[0]]
            rz_params = [torch.arctan((feature) ** 2) for feature in features[0]]
            
            # Apply Hadamard, RY, and RZ gates to each qubit
            for i in range(self.n_qubits):
                qml.Hadamard(wires=wires_type[i])
                qml.RY(ry_params[i], wires=wires_type[i])
                qml.RZ(rz_params[i], wires=wires_type[i])

            # Apply the tensor ring circuit
            self.ring_circuit(weights)

        # Define the forget gate circuit
        def _circuit_forget(inputs, weights):
            VQC(inputs, weights, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_forget]

        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        # Define the input gate circuit
        def _circuit_input(inputs, weights):
            VQC(inputs, weights, self.wires_input)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_input]

        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        # Define the update gate circuit
        def _circuit_update(inputs, weights):
            VQC(inputs, weights, self.wires_update)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_update]

        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        # Define the output gate circuit
        def _circuit_output(inputs, weights):
            VQC(inputs, weights, self.wires_output)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_output]

        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")

        # Define weight shapes for the quantum layers
        weight_shapes = {"weights": (self.num_ring_params,)}

        # Classical linear layer to map concatenated input to qubit dimension
        self.clayer_in = torch.nn.Linear(self.concat_size, self.n_qubits)
        # Quantum layers for each gate
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        # Classical linear layer to map qubit output to hidden size
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    # Forward pass of the model
    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        '''
        # Determine batch size and sequence length based on batch_first flag
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        # Initialize hidden and cell states if not provided
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        # Process each time step in the sequence
        for t in range(seq_length):
            x_t = x[:, t, :]  # Current input at time t
            v_t = torch.cat((h_t, x_t), dim=1)  # Concatenate hidden state and input
            
            # Map to qubit dimension
            y_t = self.clayer_in(v_t)

            # Compute gates using quantum circuits
            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))  # Forget gate
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))   # Input gate
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t)))     # Update gate
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t)))  # Output gate

            # Update cell state and hidden state
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        
        # Concatenate and transpose hidden sequence
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

# Define a shallow regression model using QLSTMTensorRing
class QShallowRegressionLSTMTensorRing(nn.Module):
    def __init__(self, num_sensors, hidden_units, n_qubits=4, n_qlayers=1):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = 1

        # Initialize the quantum LSTM layer
        self.lstm = QLSTMTensorRing(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers
        )

        # Linear layer for regression output
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    # Forward pass for regression
    def forward(self, x):
        batch_size = x.shape[0]
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        # Pass through LSTM and get final hidden state
        _, (hn, _) = self.lstm(x, (h0, c0))
        # Apply linear layer and flatten for output
        out = self.linear(hn).flatten()
        return out