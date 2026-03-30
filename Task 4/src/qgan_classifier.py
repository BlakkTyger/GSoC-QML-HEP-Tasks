"""
QGAN-Inspired Quantum Classifier for HEP Signal/Background Classification
=========================================================================

This module implements a Variational Quantum Classifier (VQC) using 
Google Cirq and TensorFlow Quantum for classifying High Energy Physics
events as signal or background.

Architecture follows the planning document decisions:
- Angle encoding with RY gates
- Configurable variational layers with entanglement
- Expectation value measurement
- Classical post-processing

Author: GSoC-QML-HEP Task 4
"""

import sys
import os

# Ensure we can find the installed packages
site_packages = '/home/blakktyger/.pyenv/versions/3.12.10/lib/python3.12/site-packages'
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)

import numpy as np
import cirq
import sympy
import tensorflow as tf

try:
    import tensorflow_quantum as tfq
    TFQ_AVAILABLE = True
except ImportError:
    TFQ_AVAILABLE = False
    print("WARNING: TensorFlow Quantum not available. Using simulation mode.")

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
import json


# =============================================================================
# Keras 3 Compatible PQC Layer
# =============================================================================

class Keras3PQC(tf.keras.layers.Layer):
    """Keras 3 compatible PQC layer wrapper.
    
    This layer wraps TFQ's expectation calculation to be compatible with Keras 3's
    add_weight API changes. It appends a parameterized circuit to input circuits
    and computes expectation values.
    """
    
    def __init__(self, model_circuit, operators, symbol_names, **kwargs):
        super().__init__(**kwargs)
        self.model_circuit = model_circuit
        self.operators = operators
        self.symbol_names = symbol_names
        self._num_symbols = len(symbol_names)
        # Convert circuit to tensor for appending
        self._model_circuit_tensor = tfq.convert_to_tensor([model_circuit])
        # Create append layer
        self._append_layer = tfq.layers.AddCircuit()
        
    def build(self, input_shape):
        # Use Keras 3 compatible add_weight signature
        self.symbol_values = self.add_weight(
            name='symbol_values',
            shape=(self._num_symbols,),
            initializer=tf.keras.initializers.RandomUniform(-np.pi, np.pi),
            trainable=True,
            dtype=tf.float32
        )
        super().build(input_shape)
        
    def call(self, inputs):
        # Get batch size and tile model circuit to match
        batch_size = tf.shape(inputs)[0]
        tiled_circuits = tf.tile(self._model_circuit_tensor, [batch_size])
        
        # Append the variational circuit to each input (data encoding) circuit
        appended_circuits = self._append_layer(
            inputs, 
            append=tiled_circuits
        )
        
        # Batch the symbol values for all inputs
        batch_symbols = tf.tile(
            tf.expand_dims(self.symbol_values, 0),
            [batch_size, 1]
        )
        
        # Use TFQ's expectation layer
        expectation_layer = tfq.layers.Expectation()
        return expectation_layer(
            appended_circuits,
            symbol_names=self.symbol_names,
            symbol_values=batch_symbols,
            operators=self.operators
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QGANConfig:
    """Configuration for the QGAN Classifier."""
    n_qubits: int = 5
    n_layers: int = 2
    encoding_type: str = 'angle'  # 'angle', 'angle_rz', 'iqp'
    entanglement: str = 'linear'  # 'none', 'linear', 'circular', 'full'
    learning_rate: float = 0.05
    batch_size: int = 16
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'encoding_type': self.encoding_type,
            'entanglement': self.entanglement,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience
        }


# =============================================================================
# Data Loading Module
# =============================================================================

class HEPDataLoader:
    """Load and preprocess HEP event data from NPZ file."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.train_data = None
        self.test_data = None
        
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data and return X_train, y_train, X_test, y_test."""
        data = np.load(self.filepath, allow_pickle=True)
        
        # Extract training data
        train_dict = data['training_input'].item()
        X_train_bg = train_dict['0']
        X_train_sig = train_dict['1']
        X_train = np.vstack([X_train_bg, X_train_sig])
        y_train = np.concatenate([
            np.zeros(X_train_bg.shape[0]),
            np.ones(X_train_sig.shape[0])
        ])
        
        # Extract test data
        test_dict = data['test_input'].item()
        X_test_bg = test_dict['0']
        X_test_sig = test_dict['1']
        X_test = np.vstack([X_test_bg, X_test_sig])
        y_test = np.concatenate([
            np.zeros(X_test_bg.shape[0]),
            np.ones(X_test_sig.shape[0])
        ])
        
        # Shuffle training data
        train_idx = np.random.permutation(len(y_train))
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        
        # Shuffle test data
        test_idx = np.random.permutation(len(y_test))
        X_test, y_test = X_test[test_idx], y_test[test_idx]
        
        self.train_data = (X_train, y_train)
        self.test_data = (X_test, y_test)
        
        return X_train, y_train, X_test, y_test
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self.train_data is None:
            self.load()
        
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        
        return {
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_features': X_train.shape[1],
            'train_class_dist': {
                'background': int(np.sum(y_train == 0)),
                'signal': int(np.sum(y_train == 1))
            },
            'test_class_dist': {
                'background': int(np.sum(y_test == 0)),
                'signal': int(np.sum(y_test == 1))
            },
            'feature_ranges': {
                'min': float(X_train.min()),
                'max': float(X_train.max())
            }
        }


# =============================================================================
# Quantum Circuit Module
# =============================================================================

class QuantumCircuitBuilder:
    """Build parameterized quantum circuits for classification."""
    
    def __init__(self, config: QGANConfig):
        self.config = config
        self.qubits = cirq.GridQubit.rect(1, config.n_qubits)
        self.symbols = None
        
    def _angle_encoding(self, data_symbols: List[sympy.Symbol]) -> cirq.Circuit:
        """Create angle encoding circuit using RY gates."""
        circuit = cirq.Circuit()
        for i, qubit in enumerate(self.qubits):
            circuit.append(cirq.ry(np.pi * data_symbols[i])(qubit))
        return circuit
    
    def _angle_rz_encoding(self, data_symbols: List[sympy.Symbol]) -> cirq.Circuit:
        """Create enhanced angle encoding with RY + RZ gates."""
        circuit = cirq.Circuit()
        for i, qubit in enumerate(self.qubits):
            circuit.append(cirq.ry(np.pi * data_symbols[i])(qubit))
            circuit.append(cirq.rz(np.pi * data_symbols[i])(qubit))
        return circuit
    
    def _iqp_encoding(self, data_symbols: List[sympy.Symbol]) -> cirq.Circuit:
        """Create IQP-style encoding with interactions."""
        circuit = cirq.Circuit()
        # Hadamard layer
        for qubit in self.qubits:
            circuit.append(cirq.H(qubit))
        # Z rotations
        for i, qubit in enumerate(self.qubits):
            circuit.append(cirq.rz(np.pi * data_symbols[i])(qubit))
        # ZZ interactions
        for i in range(len(self.qubits) - 1):
            circuit.append(cirq.ZZ(self.qubits[i], self.qubits[i+1])**(
                data_symbols[i] * data_symbols[i+1] / np.pi
            ))
        return circuit
    
    def _variational_layer(self, layer_symbols: List[sympy.Symbol], layer_idx: int) -> cirq.Circuit:
        """Create a single variational layer."""
        circuit = cirq.Circuit()
        n = self.config.n_qubits
        
        # Single-qubit rotations: RY and RZ for each qubit
        for i, qubit in enumerate(self.qubits):
            circuit.append(cirq.ry(layer_symbols[2*i])(qubit))
            circuit.append(cirq.rz(layer_symbols[2*i + 1])(qubit))
        
        # Entangling gates based on configuration
        if self.config.entanglement == 'linear':
            for i in range(n - 1):
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i+1]))
        elif self.config.entanglement == 'circular':
            for i in range(n):
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[(i+1) % n]))
        elif self.config.entanglement == 'full':
            for i in range(n):
                for j in range(i+1, n):
                    circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
        # 'none' - no entanglement
        
        return circuit
    
    def build_encoding_circuit(self) -> Tuple[cirq.Circuit, List[sympy.Symbol]]:
        """Build the data encoding circuit."""
        data_symbols = [sympy.Symbol(f'x_{i}') for i in range(self.config.n_qubits)]
        
        if self.config.encoding_type == 'angle':
            circuit = self._angle_encoding(data_symbols)
        elif self.config.encoding_type == 'angle_rz':
            circuit = self._angle_rz_encoding(data_symbols)
        elif self.config.encoding_type == 'iqp':
            circuit = self._iqp_encoding(data_symbols)
        else:
            raise ValueError(f"Unknown encoding type: {self.config.encoding_type}")
        
        return circuit, data_symbols
    
    def build_variational_circuit(self) -> Tuple[cirq.Circuit, List[sympy.Symbol]]:
        """Build the variational (trainable) circuit."""
        all_symbols = []
        circuit = cirq.Circuit()
        
        params_per_layer = 2 * self.config.n_qubits  # RY + RZ for each qubit
        
        for layer in range(self.config.n_layers):
            layer_symbols = [
                sympy.Symbol(f'theta_{layer}_{i}') 
                for i in range(params_per_layer)
            ]
            all_symbols.extend(layer_symbols)
            circuit += self._variational_layer(layer_symbols, layer)
        
        self.symbols = all_symbols
        return circuit, all_symbols
    
    def build_full_circuit(self) -> Tuple[cirq.Circuit, List[sympy.Symbol], List[sympy.Symbol]]:
        """Build complete circuit: encoding + variational."""
        encoding_circuit, data_symbols = self.build_encoding_circuit()
        variational_circuit, var_symbols = self.build_variational_circuit()
        
        full_circuit = encoding_circuit + variational_circuit
        return full_circuit, data_symbols, var_symbols
    
    def get_readout_operators(self) -> List[cirq.PauliString]:
        """Get measurement operators."""
        # Measure Z expectation on first qubit
        return [cirq.Z(self.qubits[0])]
    
    def print_circuit_info(self):
        """Print circuit information."""
        _, data_sym, var_sym = self.build_full_circuit()
        print(f"Circuit Configuration:")
        print(f"  - Qubits: {self.config.n_qubits}")
        print(f"  - Encoding: {self.config.encoding_type}")
        print(f"  - Variational Layers: {self.config.n_layers}")
        print(f"  - Entanglement: {self.config.entanglement}")
        print(f"  - Data parameters: {len(data_sym)}")
        print(f"  - Trainable parameters: {len(var_sym)}")


# =============================================================================
# Data Encoding for TFQ
# =============================================================================

class QuantumDataEncoder:
    """Encode classical data into quantum circuits for TFQ."""
    
    def __init__(self, circuit_builder: QuantumCircuitBuilder):
        self.circuit_builder = circuit_builder
        self.encoding_circuit, self.data_symbols = circuit_builder.build_encoding_circuit()
        
    def encode_data_point(self, x: np.ndarray) -> cirq.Circuit:
        """Encode a single data point into a quantum circuit."""
        resolver = {self.data_symbols[i]: x[i] for i in range(len(x))}
        return cirq.resolve_parameters(self.encoding_circuit, resolver)
    
    def encode_batch(self, X: np.ndarray) -> List[cirq.Circuit]:
        """Encode a batch of data points."""
        return [self.encode_data_point(x) for x in X]
    
    def encode_to_tfq_tensor(self, X: np.ndarray) -> tf.Tensor:
        """Encode data and convert to TFQ tensor."""
        if not TFQ_AVAILABLE:
            raise RuntimeError("TensorFlow Quantum is required for this operation")
        circuits = self.encode_batch(X)
        return tfq.convert_to_tensor(circuits)


# =============================================================================
# Model Building
# =============================================================================

class QGANClassifier:
    """Main QGAN-inspired quantum classifier."""
    
    def __init__(self, config: QGANConfig):
        self.config = config
        self.circuit_builder = QuantumCircuitBuilder(config)
        self.encoder = QuantumDataEncoder(self.circuit_builder)
        self.model = None
        self.history = None
        
    def build_model(self) -> tf.keras.Model:
        """Build the Keras model with TFQ layers."""
        if not TFQ_AVAILABLE:
            raise RuntimeError("TensorFlow Quantum is required")
        
        # Build variational circuit
        var_circuit, var_symbols = self.circuit_builder.build_variational_circuit()
        readout_ops = self.circuit_builder.get_readout_operators()
        
        # Get symbol names as strings for TFQ
        symbol_names = [str(s) for s in var_symbols]
        
        # Build model using Keras 3 compatible PQC wrapper
        # The quantum layer outputs expectation values in [-1, 1]
        # We add hidden layers to learn the optimal mapping to classification
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            Keras3PQC(var_circuit, readout_ops, symbol_names),
            # Rescale from [-1, 1] to [0, 1] for better gradient flow
            tf.keras.layers.Lambda(lambda x: (x + 1) / 2),
            # Add hidden layer for more expressive power
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[tf.Tensor, np.ndarray]:
        """Prepare data for training."""
        X_encoded = self.encoder.encode_to_tfq_tensor(X)
        y = y.astype(np.float32)
        return X_encoded, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              verbose: int = 1) -> tf.keras.callbacks.History:
        """Train the model."""
        if self.model is None:
            self.build_model()
        
        # Prepare training data
        X_train_enc, y_train = self.prepare_data(X_train, y_train)
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_enc, y_val = self.prepare_data(X_val, y_val)
            validation_data = (X_val_enc, y_val)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train_enc, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for input data."""
        X_enc = self.encoder.encode_to_tfq_tensor(X)
        return self.model.predict(X_enc, verbose=0).flatten()
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred_probs = self.predict(X)
        y_pred = (y_pred_probs >= 0.5).astype(int)
        
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_probs)
        
        return {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'predictions': y_pred_probs
        }


# =============================================================================
# Classical Baseline (for comparison)
# =============================================================================

class ClassicalBaseline:
    """Classical neural network baseline for comparison."""
    
    def __init__(self, n_features: int, hidden_units: List[int] = [16, 8]):
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.model = None
        
    def build_model(self) -> tf.keras.Model:
        """Build classical neural network."""
        layers = [tf.keras.layers.Input(shape=(self.n_features,))]
        
        for units in self.hidden_units:
            layers.append(tf.keras.layers.Dense(units, activation='relu'))
        
        layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        self.model = tf.keras.Sequential(layers)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              validation_split: float = 0.2, epochs: int = 100,
              verbose: int = 1) -> tf.keras.callbacks.History:
        """Train the classical model."""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
        ]
        
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model."""
        y_pred_probs = self.model.predict(X, verbose=0).flatten()
        y_pred = (y_pred_probs >= 0.5).astype(int)
        
        return {
            'accuracy': float(accuracy_score(y, y_pred)),
            'auc': float(roc_auc_score(y, y_pred_probs))
        }


# =============================================================================
# Training Pipeline
# =============================================================================

class TrainingPipeline:
    """Complete training pipeline with experiment tracking."""
    
    def __init__(self, data_path: str, results_dir: str = 'results'):
        self.data_path = data_path
        self.results_dir = results_dir
        self.loader = HEPDataLoader(data_path)
        self.results = []
        
        os.makedirs(results_dir, exist_ok=True)
        
    def run_experiment(self, config: QGANConfig, experiment_name: str = None) -> Dict[str, Any]:
        """Run a single training experiment."""
        if experiment_name is None:
            experiment_name = f"exp_{len(self.results)}"
        
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*60}")
        
        # Load data
        X_train, y_train, X_test, y_test = self.loader.load()
        
        # Create validation split from training data
        n_val = int(len(y_train) * config.validation_split)
        X_val, y_val = X_train[:n_val], y_train[:n_val]
        X_train_split, y_train_split = X_train[n_val:], y_train[n_val:]
        
        print(f"Training samples: {len(y_train_split)}")
        print(f"Validation samples: {len(y_val)}")
        print(f"Test samples: {len(y_test)}")
        
        # Build and train model
        classifier = QGANClassifier(config)
        classifier.circuit_builder.print_circuit_info()
        
        print("\nTraining...")
        history = classifier.train(X_train_split, y_train_split, X_val, y_val, verbose=1)
        
        # Evaluate
        print("\nEvaluating...")
        train_results = classifier.evaluate(X_train_split, y_train_split)
        val_results = classifier.evaluate(X_val, y_val)
        test_results = classifier.evaluate(X_test, y_test)
        
        print(f"\nResults:")
        print(f"  Train - Accuracy: {train_results['accuracy']:.4f}, AUC: {train_results['auc']:.4f}")
        print(f"  Val   - Accuracy: {val_results['accuracy']:.4f}, AUC: {val_results['auc']:.4f}")
        print(f"  Test  - Accuracy: {test_results['accuracy']:.4f}, AUC: {test_results['auc']:.4f}")
        
        # Store results
        result = {
            'name': experiment_name,
            'config': config.to_dict(),
            'train': train_results,
            'val': val_results,
            'test': test_results,
            'epochs_trained': len(history.history['loss'])
        }
        
        # Remove predictions from stored results (too large)
        for split in ['train', 'val', 'test']:
            if 'predictions' in result[split]:
                del result[split]['predictions']
        
        self.results.append(result)
        
        return result
    
    def run_hyperparameter_search(self) -> List[Dict[str, Any]]:
        """Run systematic hyperparameter search."""
        search_space = {
            'n_layers': [1, 2, 3],
            'encoding_type': ['angle', 'angle_rz'],
            'entanglement': ['linear', 'circular'],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        
        # Generate configurations (subset for efficiency)
        configs = [
            # Vary layers
            QGANConfig(n_layers=1, encoding_type='angle', entanglement='linear', learning_rate=0.05),
            QGANConfig(n_layers=2, encoding_type='angle', entanglement='linear', learning_rate=0.05),
            QGANConfig(n_layers=3, encoding_type='angle', entanglement='linear', learning_rate=0.05),
            # Vary encoding
            QGANConfig(n_layers=2, encoding_type='angle_rz', entanglement='linear', learning_rate=0.05),
            # Vary entanglement
            QGANConfig(n_layers=2, encoding_type='angle', entanglement='circular', learning_rate=0.05),
            QGANConfig(n_layers=2, encoding_type='angle', entanglement='none', learning_rate=0.05),
            # Vary learning rate
            QGANConfig(n_layers=2, encoding_type='angle', entanglement='linear', learning_rate=0.01),
            QGANConfig(n_layers=2, encoding_type='angle', entanglement='linear', learning_rate=0.1),
        ]
        
        for i, config in enumerate(configs):
            name = f"hp_search_{i}_L{config.n_layers}_{config.encoding_type}_{config.entanglement}_lr{config.learning_rate}"
            self.run_experiment(config, name)
        
        return self.results
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the best configuration based on test AUC."""
        if not self.results:
            raise ValueError("No experiments run yet")
        
        best = max(self.results, key=lambda x: x['test']['auc'])
        return best
    
    def save_results(self, filename: str = 'experiment_results.json'):
        """Save all results to JSON."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def print_summary(self):
        """Print summary of all experiments."""
        if not self.results:
            print("No experiments run yet")
            return
        
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"{'Name':<40} {'Test Acc':>10} {'Test AUC':>10} {'Epochs':>8}")
        print("-"*80)
        
        for r in sorted(self.results, key=lambda x: x['test']['auc'], reverse=True):
            print(f"{r['name']:<40} {r['test']['accuracy']:>10.4f} {r['test']['auc']:>10.4f} {r['epochs_trained']:>8}")
        
        best = self.get_best_config()
        print("-"*80)
        print(f"Best configuration: {best['name']}")
        print(f"Test Accuracy: {best['test']['accuracy']:.4f}")
        print(f"Test AUC: {best['test']['auc']:.4f}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Paths
    data_path = "/home/blakktyger/Documents/BlakkTyger/Projects/GSOC-26/GSoC-QML-HEP-Tasks/Task 4/QIS_EXAM_200Events.npz"
    results_dir = "/home/blakktyger/Documents/BlakkTyger/Projects/GSOC-26/GSoC-QML-HEP-Tasks/Task 4/results"
    
    # Check TFQ availability
    if not TFQ_AVAILABLE:
        print("ERROR: TensorFlow Quantum is not available.")
        print("Please install with: pip install tensorflow-quantum")
        print("\nRunning in demo mode with circuit visualization only...")
        
        # Demo: Show circuit structure
        config = QGANConfig()
        builder = QuantumCircuitBuilder(config)
        full_circuit, data_sym, var_sym = builder.build_full_circuit()
        
        print("\n" + "="*60)
        print("CIRCUIT STRUCTURE")
        print("="*60)
        print(full_circuit)
        print(f"\nData symbols: {data_sym}")
        print(f"Variational symbols: {var_sym[:6]}... ({len(var_sym)} total)")
        return
    
    # Create pipeline
    pipeline = TrainingPipeline(data_path, results_dir)
    
    # Print dataset statistics
    print("Dataset Statistics:")
    stats = pipeline.loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Run hyperparameter search
    print("\n" + "="*60)
    print("STARTING HYPERPARAMETER SEARCH")
    print("="*60)
    
    pipeline.run_hyperparameter_search()
    
    # Print summary
    pipeline.print_summary()
    
    # Save results
    pipeline.save_results()
    
    # Get best configuration for final report
    best = pipeline.get_best_config()
    print(f"\nBest Configuration:")
    print(json.dumps(best['config'], indent=2))


if __name__ == "__main__":
    main()
