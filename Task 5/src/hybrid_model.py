"""
Hybrid Classical-Quantum Model for jet classification.
Combines classical preprocessing with quantum circuit and classical readout.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import cirq
from .qgnn_circuit import QGNNCircuit, get_expectation_values
from .preprocessing import preprocess_jet


class HybridQGNNClassifier:
    """
    Hybrid Quantum-Classical GNN classifier for jet tagging.
    
    Architecture:
    1. Classical preprocessing: Select particles, normalize features
    2. Quantum circuit: Graph-structured QGNN
    3. Classical readout: MLP for final classification
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        k_neighbors: int = 4,
        learning_rate: float = 0.01
    ):
        """
        Initialize hybrid model.
        
        Args:
            n_qubits: Number of qubits (particles to select)
            n_layers: Number of QGNN layers
            k_neighbors: k for k-NN graph
            learning_rate: Learning rate for optimization
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.k_neighbors = k_neighbors
        self.learning_rate = learning_rate
        
        # QGNN circuit builder
        self.qgnn = QGNNCircuit(n_qubits=n_qubits, n_layers=n_layers)
        
        # Initialize trainable parameters
        self.n_quantum_params = self.qgnn.get_num_params()
        self.quantum_params = np.random.uniform(
            -np.pi, np.pi, self.n_quantum_params
        )
        
        # Classical readout weights
        # Input: n_qubits expectation values, Output: 2 classes
        self.hidden_dim = 16
        self.W1 = np.random.randn(n_qubits, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, 2) * 0.1
        self.b2 = np.zeros(2)
        
        # Training history
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    def _preprocess(self, jet: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess a single jet."""
        return preprocess_jet(jet, self.n_qubits, self.k_neighbors)
    
    def _build_param_dict(self) -> Dict:
        """Build parameter dictionary for circuit evaluation."""
        symbols = self.qgnn.get_symbols()
        return {symbols[i]: self.quantum_params[i] for i in range(len(symbols))}
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _classical_readout(self, expectations: np.ndarray) -> np.ndarray:
        """
        Classical MLP readout.
        
        Args:
            expectations: (n_qubits,) Z expectation values
            
        Returns:
            (2,) class probabilities
        """
        # Hidden layer
        h = self._relu(expectations @ self.W1 + self.b1)
        # Output layer
        logits = h @ self.W2 + self.b2
        return self._softmax(logits)
    
    def forward(self, jet: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through hybrid model.
        
        Args:
            jet: (M, 4) jet particle array
            
        Returns:
            probs: (2,) class probabilities
            expectations: (n_qubits,) intermediate expectations
        """
        # Preprocess
        features, edge_index = self._preprocess(jet)
        
        # Build circuit
        circuit = self.qgnn.build_circuit(features, edge_index)
        
        # Get expectation values
        observables = self.qgnn.get_observables()
        param_dict = self._build_param_dict()
        expectations = get_expectation_values(circuit, observables, param_dict)
        
        # Classical readout
        probs = self._classical_readout(expectations)
        
        return probs, expectations
    
    def predict(self, jet: np.ndarray) -> int:
        """Predict class for a single jet."""
        probs, _ = self.forward(jet)
        return np.argmax(probs)
    
    def predict_batch(self, jets: np.ndarray) -> np.ndarray:
        """Predict classes for batch of jets."""
        predictions = []
        for jet in jets:
            predictions.append(self.predict(jet))
        return np.array(predictions)
    
    def compute_loss(self, probs: np.ndarray, label: int) -> float:
        """Cross-entropy loss."""
        eps = 1e-10
        return -np.log(probs[label] + eps)
    
    def _parameter_shift_gradient(
        self,
        jet: np.ndarray,
        label: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients using parameter-shift rule (more efficient for quantum).
        
        Args:
            jet: Input jet
            label: True label
            
        Returns:
            Gradients for quantum params and W2
        """
        shift = np.pi / 2
        
        # Gradient for quantum parameters using parameter-shift
        grad_quantum = np.zeros_like(self.quantum_params)
        for i in range(len(self.quantum_params)):
            # Shift +
            self.quantum_params[i] += shift
            probs_plus, _ = self.forward(jet)
            loss_plus = self.compute_loss(probs_plus, label)
            
            # Shift -
            self.quantum_params[i] -= 2 * shift
            probs_minus, _ = self.forward(jet)
            loss_minus = self.compute_loss(probs_minus, label)
            
            # Restore
            self.quantum_params[i] += shift
            
            grad_quantum[i] = (loss_plus - loss_minus) / 2
        
        # Analytical gradient for classical output layer
        probs, expectations = self.forward(jet)
        h = self._relu(expectations @ self.W1 + self.b1)
        
        # Gradient of cross-entropy w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[label] -= 1
        
        # Gradient for W2
        grad_W2 = np.outer(h, grad_logits)
        
        return grad_quantum, grad_W2
    
    def train_step(self, jet: np.ndarray, label: int) -> float:
        """
        Single training step with gradient descent.
        
        Args:
            jet: Input jet
            label: True label
            
        Returns:
            Loss value
        """
        # Compute loss before update
        probs, _ = self.forward(jet)
        loss = self.compute_loss(probs, int(label))
        
        # Compute gradients using parameter-shift
        grad_q, grad_W2 = self._parameter_shift_gradient(jet, int(label))
        
        # Gradient clipping
        grad_q = np.clip(grad_q, -1.0, 1.0)
        grad_W2 = np.clip(grad_W2, -1.0, 1.0)
        
        # Update parameters
        self.quantum_params -= self.learning_rate * grad_q
        self.W2 -= self.learning_rate * grad_W2
        
        return loss
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training jets
            y_train: Training labels
            X_val: Validation jets
            y_val: Validation labels
            epochs: Number of epochs
            verbose: Print progress
            
        Returns:
            Training history
        """
        n_train = len(X_train)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_train)
            
            epoch_loss = 0
            correct = 0
            
            for idx in indices:
                jet = X_train[idx]
                label = int(y_train[idx])
                
                # Training step
                loss = self.train_step(jet, label)
                epoch_loss += loss
                
                # Accuracy
                pred = self.predict(jet)
                if pred == label:
                    correct += 1
            
            # Record metrics
            train_loss = epoch_loss / n_train
            train_acc = correct / n_train
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
            else:
                val_loss, val_acc = None, None
            
            if verbose:
                val_str = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" if val_loss else ""
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}{val_str}")
        
        return self.history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model on dataset.
        
        Returns:
            (loss, accuracy)
        """
        total_loss = 0
        correct = 0
        
        for jet, label in zip(X, y):
            probs, _ = self.forward(jet)
            total_loss += self.compute_loss(probs, int(label))
            if np.argmax(probs) == int(label):
                correct += 1
        
        return total_loss / len(X), correct / len(X)
    
    def get_circuit_for_jet(self, jet: np.ndarray) -> cirq.Circuit:
        """Get the quantum circuit for a specific jet."""
        features, edge_index = self._preprocess(jet)
        return self.qgnn.build_circuit(features, edge_index)


class SimpleQGNNClassifier:
    """
    Simplified QGNN classifier using only quantum circuit output.
    For demonstration and testing purposes.
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        k_neighbors: int = 4
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.k_neighbors = k_neighbors
        
        self.qgnn = QGNNCircuit(n_qubits=n_qubits, n_layers=n_layers)
        self.quantum_params = np.random.uniform(-np.pi, np.pi, self.qgnn.get_num_params())
    
    def forward(self, jet: np.ndarray) -> float:
        """
        Forward pass returning single expectation value for classification.
        
        Returns sum of Z expectations, positive → quark, negative → gluon
        """
        features, edge_index = preprocess_jet(jet, self.n_qubits, self.k_neighbors)
        circuit = self.qgnn.build_circuit(features, edge_index)
        
        observables = self.qgnn.get_observables()
        symbols = self.qgnn.get_symbols()
        param_dict = {symbols[i]: self.quantum_params[i] for i in range(len(symbols))}
        
        expectations = get_expectation_values(circuit, observables, param_dict)
        return np.mean(expectations)
    
    def predict(self, jet: np.ndarray) -> int:
        """Predict: 1 (quark) if expectation > 0, else 0 (gluon)."""
        return 1 if self.forward(jet) > 0 else 0
