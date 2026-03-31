"""
Quantum Neural Network models for Z₂ × Z₂ symmetric classification.

Implements:
1. Standard QNN (no symmetry constraints)
2. Equivariant QNN (respects Z₂ × Z₂ symmetry via tied parameters)
"""

import pennylane as qml
import torch
import torch.nn as nn
from typing import Dict


N_QUBITS = 2
N_LAYERS = 3


def create_standard_qnn() -> qml.qnn.TorchLayer:
    """
    Create a standard (non-equivariant) QNN.
    
    Architecture:
    - AngleEmbedding for data encoding
    - BasicEntanglerLayers with independent parameters
    - Measure Z on all qubits
    """
    dev = qml.device("default.qubit", wires=N_QUBITS)
    
    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    
    weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


def create_equivariant_qnn() -> qml.qnn.TorchLayer:
    """
    Create a Z₂ × Z₂ equivariant QNN.
    
    Architecture:
    - AngleEmbedding for data encoding
    - Equivariant layers with TIED parameters (same rotation on both qubits)
    - CNOT for entanglement
    - Measure Z on all qubits
    
    Equivariance is achieved by using the same parameter for both qubits.
    """
    dev = qml.device("default.qubit", wires=N_QUBITS)
    
    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        
        n_layers = weights.shape[0]
        for i in range(n_layers):
            qml.RX(weights[i, 0], wires=0)
            qml.RX(weights[i, 0], wires=1)
            qml.CNOT(wires=[0, 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    
    weight_shapes = {"weights": (N_LAYERS, 1)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QNNClassifier(nn.Module):
    """
    Hybrid quantum-classical classifier.
    
    Combines a QNN layer with a classical linear layer for classification.
    """
    
    def __init__(self, qnn_layer: qml.qnn.TorchLayer, name: str = "QNN"):
        super().__init__()
        self.name = name
        self.qnn = qnn_layer
        self.fc = nn.Linear(N_QUBITS, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.qnn(x)
        logits = self.fc(q_out)
        return nn.functional.log_softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_standard_classifier() -> QNNClassifier:
    """Create standard QNN classifier."""
    return QNNClassifier(create_standard_qnn(), name="Standard QNN")


def create_equivariant_classifier() -> QNNClassifier:
    """Create equivariant QNN classifier."""
    return QNNClassifier(create_equivariant_qnn(), name="Equivariant QNN")


if __name__ == "__main__":
    std_model = create_standard_classifier()
    eqv_model = create_equivariant_classifier()
    
    print(f"Standard QNN parameters: {std_model.count_parameters()}")
    print(f"Equivariant QNN parameters: {eqv_model.count_parameters()}")
    
    x_test = torch.rand(5, 2)
    print(f"\nStandard QNN output shape: {std_model(x_test).shape}")
    print(f"Equivariant QNN output shape: {eqv_model(x_test).shape}")
