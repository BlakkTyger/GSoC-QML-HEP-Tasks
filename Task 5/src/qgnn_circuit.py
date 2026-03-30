"""
Quantum Graph Neural Network circuit implementation using Cirq.
Implements a hybrid data re-uploading QGNN with graph-structured entanglement.
"""

import cirq
import sympy
import numpy as np
from typing import List, Tuple, Optional


class QGNNCircuit:
    """
    Quantum Graph Neural Network circuit builder.
    
    Architecture:
    - Data re-uploading: Features encoded multiple times
    - Graph-structured entanglement: CZ gates on k-NN edges
    - Variational rotations: Trainable Ry, Rz gates
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        n_features: int = 3
    ):
        """
        Initialize QGNN circuit.
        
        Args:
            n_qubits: Number of qubits (one per particle)
            n_layers: Number of variational layers
            n_features: Number of features per particle
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        
        # Create qubits
        self.qubits = cirq.LineQubit.range(n_qubits)
        
        # Create symbols for variational parameters
        self.symbols = self._create_symbols()
        
    def _create_symbols(self) -> List[sympy.Symbol]:
        """Create symbolic parameters for the circuit."""
        symbols = []
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                # Two variational parameters per qubit per layer (Ry, Rz)
                symbols.append(sympy.Symbol(f'theta_{layer}_{qubit}_y'))
                symbols.append(sympy.Symbol(f'theta_{layer}_{qubit}_z'))
        return symbols
    
    def get_symbols(self) -> List[sympy.Symbol]:
        """Return list of trainable symbols."""
        return self.symbols
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return len(self.symbols)
    
    def _encoding_layer(
        self,
        features: np.ndarray,
        circuit: cirq.Circuit
    ) -> cirq.Circuit:
        """
        Add feature encoding layer.
        
        Encodes features as rotation angles:
        - Rx(feature_0): Δη
        - Ry(feature_1): Δφ  
        - Rz(feature_2): log_pT
        
        Args:
            features: (n_qubits, n_features) feature array
            circuit: Circuit to add gates to
            
        Returns:
            Updated circuit
        """
        for i, qubit in enumerate(self.qubits):
            if i < len(features):
                # Encode each feature as a rotation
                circuit.append(cirq.rx(features[i, 0])(qubit))
                circuit.append(cirq.ry(features[i, 1])(qubit))
                if self.n_features > 2:
                    circuit.append(cirq.rz(features[i, 2])(qubit))
        return circuit
    
    def _variational_layer(
        self,
        layer_idx: int,
        circuit: cirq.Circuit
    ) -> cirq.Circuit:
        """
        Add variational rotation layer with trainable parameters.
        
        Args:
            layer_idx: Index of current layer
            circuit: Circuit to add gates to
            
        Returns:
            Updated circuit
        """
        for i, qubit in enumerate(self.qubits):
            param_idx = layer_idx * self.n_qubits * 2 + i * 2
            theta_y = self.symbols[param_idx]
            theta_z = self.symbols[param_idx + 1]
            
            circuit.append(cirq.ry(theta_y)(qubit))
            circuit.append(cirq.rz(theta_z)(qubit))
        
        return circuit
    
    def _entanglement_layer(
        self,
        edge_index: np.ndarray,
        circuit: cirq.Circuit
    ) -> cirq.Circuit:
        """
        Add graph-structured entanglement layer.
        
        Uses CZ gates on edges defined by the k-NN graph.
        
        Args:
            edge_index: (2, E) array of edge indices
            circuit: Circuit to add gates to
            
        Returns:
            Updated circuit
        """
        # Get unique edges (avoid duplicate CZ for undirected graph)
        processed_edges = set()
        
        for e in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, e]), int(edge_index[1, e])
            
            # Skip if already processed (undirected)
            edge_key = (min(src, dst), max(src, dst))
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)
            
            # Ensure valid qubit indices
            if src < self.n_qubits and dst < self.n_qubits and src != dst:
                circuit.append(cirq.CZ(self.qubits[src], self.qubits[dst]))
        
        return circuit
    
    def build_circuit(
        self,
        features: np.ndarray,
        edge_index: np.ndarray
    ) -> cirq.Circuit:
        """
        Build the full QGNN circuit.
        
        Structure per layer:
        1. Feature encoding (Rx, Ry, Rz with data)
        2. Variational rotations (Ry, Rz with trainable params)
        3. Graph entanglement (CZ on edges)
        
        Args:
            features: (n_qubits, n_features) feature array
            edge_index: (2, E) edge indices
            
        Returns:
            Complete cirq.Circuit
        """
        circuit = cirq.Circuit()
        
        for layer in range(self.n_layers):
            # Encoding layer (data re-uploading)
            circuit = self._encoding_layer(features, circuit)
            
            # Variational layer
            circuit = self._variational_layer(layer, circuit)
            
            # Entanglement layer (graph structure)
            circuit = self._entanglement_layer(edge_index, circuit)
        
        return circuit
    
    def build_parametrized_circuit(
        self,
        edge_index: np.ndarray
    ) -> Tuple[cirq.Circuit, List[sympy.Symbol]]:
        """
        Build circuit with symbolic feature placeholders.
        
        Used for TFQ integration where features are provided at runtime.
        
        Args:
            edge_index: (2, E) edge indices
            
        Returns:
            circuit: Circuit with symbolic parameters
            feature_symbols: List of feature symbols
        """
        circuit = cirq.Circuit()
        feature_symbols = []
        
        # Create feature symbols
        for i in range(self.n_qubits):
            for f in range(self.n_features):
                feature_symbols.append(sympy.Symbol(f'x_{i}_{f}'))
        
        for layer in range(self.n_layers):
            # Encoding layer with symbolic features
            for i, qubit in enumerate(self.qubits):
                feat_idx = i * self.n_features
                circuit.append(cirq.rx(feature_symbols[feat_idx])(qubit))
                circuit.append(cirq.ry(feature_symbols[feat_idx + 1])(qubit))
                if self.n_features > 2:
                    circuit.append(cirq.rz(feature_symbols[feat_idx + 2])(qubit))
            
            # Variational layer
            circuit = self._variational_layer(layer, circuit)
            
            # Entanglement layer
            circuit = self._entanglement_layer(edge_index, circuit)
        
        return circuit, feature_symbols
    
    def get_observables(self) -> List[cirq.PauliString]:
        """
        Get measurement observables (Z on each qubit).
        
        Returns:
            List of Z observables
        """
        return [cirq.Z(q) for q in self.qubits]


def create_qgnn_circuit(
    n_qubits: int = 8,
    n_layers: int = 3,
    features: Optional[np.ndarray] = None,
    edge_index: Optional[np.ndarray] = None
) -> Tuple[cirq.Circuit, QGNNCircuit]:
    """
    Convenience function to create a QGNN circuit.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        features: Optional feature array
        edge_index: Optional edge index
        
    Returns:
        circuit: Built circuit (or empty if no features)
        qgnn: QGNNCircuit instance
    """
    qgnn = QGNNCircuit(n_qubits=n_qubits, n_layers=n_layers)
    
    if features is not None and edge_index is not None:
        circuit = qgnn.build_circuit(features, edge_index)
    else:
        circuit = cirq.Circuit()
    
    return circuit, qgnn


def simulate_circuit(
    circuit: cirq.Circuit,
    param_values: Optional[dict] = None,
    repetitions: int = 1000
) -> np.ndarray:
    """
    Simulate circuit and return measurement results.
    
    Args:
        circuit: Circuit to simulate
        param_values: Dictionary mapping symbols to values
        repetitions: Number of measurement repetitions
        
    Returns:
        Measurement results array
    """
    simulator = cirq.Simulator()
    
    if param_values:
        resolved = cirq.resolve_parameters(circuit, param_values)
    else:
        resolved = circuit
    
    # Add measurements
    qubits = sorted(circuit.all_qubits())
    measured_circuit = resolved + cirq.measure(*qubits, key='result')
    
    result = simulator.run(measured_circuit, repetitions=repetitions)
    
    return result.measurements['result']


def get_expectation_values(
    circuit: cirq.Circuit,
    observables: List[cirq.PauliString],
    param_values: Optional[dict] = None
) -> np.ndarray:
    """
    Compute expectation values of observables.
    
    Args:
        circuit: Circuit to simulate
        observables: List of Pauli observables
        param_values: Dictionary mapping symbols to values
        
    Returns:
        Array of expectation values
    """
    simulator = cirq.Simulator()
    
    if param_values:
        resolved = cirq.resolve_parameters(circuit, param_values)
    else:
        resolved = circuit
    
    # Get final state
    result = simulator.simulate(resolved)
    state = result.final_state_vector
    
    expectations = []
    for obs in observables:
        # Compute <ψ|O|ψ>
        exp_val = obs.expectation_from_state_vector(
            state, 
            qubit_map={q: i for i, q in enumerate(sorted(circuit.all_qubits()))}
        )
        expectations.append(np.real(exp_val))
    
    return np.array(expectations)
