"""Hybrid quantum-classical model for PQC embedding."""

import torch
import torch.nn as nn
import pennylane as qml


N_QUBITS = 5
N_LAYERS = 3
N_PQC_PARAMS = N_QUBITS * N_LAYERS


def create_quantum_circuit(n_qubits: int = N_QUBITS):
    """
    Create a parameterized quantum circuit.
    
    Args:
        n_qubits: Number of qubits
    
    Returns:
        QNode for the quantum circuit
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def quantum_circuit(params):
        """
        Parameterized quantum circuit with multiple layers.
        Each layer applies RY rotations on all qubits and ring CNOT entanglers.
        
        Args:
            params: Tensor of shape (n_layers, n_qubits)
        
        Returns:
            List of Pauli-Z expectation values for first 4 qubits
        """
        n_layers = params.shape[0]
        
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                qml.RY(params[layer, qubit], wires=qubit)
            for qubit in range(n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
        
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]
    
    return quantum_circuit


class HybridModel(nn.Module):
    """
    Hybrid quantum-classical model.
    
    MLP maps input to PQC parameters, which are used to generate
    quantum states. Output scaling layer maps expectation values
    to target range.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: list = [32, 64],
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        output_dim: int = 4
    ):
        """
        Initialize the hybrid model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            n_qubits: Number of qubits in PQC
            n_layers: Number of PQC layers
            output_dim: Output dimension (number of targets)
        """
        super(HybridModel, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_pqc_params = n_qubits * n_layers
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.n_pqc_params))
        
        self.mlp = nn.Sequential(*layers)
        
        self.qnode = create_quantum_circuit(n_qubits)
        
        self.output_scale = nn.Parameter(torch.ones(output_dim))
        self.output_bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape (batch, 1)
        
        Returns:
            Output tensor of shape (batch, 4)
        """
        params = self.mlp(x)
        
        outputs = []
        for i in range(x.shape[0]):
            param_sample = params[i].reshape(self.n_layers, self.n_qubits)
            qc_out = self.qnode(param_sample)
            outputs.append(torch.stack(qc_out))
        
        pqc_output = torch.stack(outputs)
        
        scaled_output = pqc_output * self.output_scale + self.output_bias
        
        return scaled_output
