"""Parameterized Quantum Circuit for PQC embedding."""

import torch
import pennylane as qml

N_QUBITS = 5
N_LAYERS = 3
N_PQC_PARAMS = N_QUBITS * N_LAYERS
OUTPUT_DIM = 4


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
        
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(OUTPUT_DIM)]
    
    return quantum_circuit


_qnode = None

def get_qnode():
    """Get or create the quantum circuit QNode (singleton)."""
    global _qnode
    if _qnode is None:
        _qnode = create_quantum_circuit(N_QUBITS)
    return _qnode


def run_pqc(params: torch.Tensor) -> torch.Tensor:
    """
    Run the PQC with given parameters.
    
    Args:
        params: Tensor of shape (15,) - flattened PQC parameters
    
    Returns:
        Tensor of shape (4,) - Pauli-Z expectation values
    """
    qnode = get_qnode()
    params_reshaped = params.reshape(N_LAYERS, N_QUBITS)
    outputs = qnode(params_reshaped)
    return torch.stack(outputs)


def compute_target(x: torch.Tensor) -> torch.Tensor:
    """
    Compute target values for input x.
    Target = [x, sin(x), cos(x), x²]
    
    Args:
        x: Input tensor of shape (1,) or scalar
    
    Returns:
        Target tensor of shape (4,)
    """
    x_val = x.squeeze()
    return torch.stack([
        x_val,
        torch.sin(x_val),
        torch.cos(x_val),
        x_val ** 2
    ])
