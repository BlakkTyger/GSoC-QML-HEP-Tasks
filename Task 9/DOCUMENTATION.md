# Task IX: Kolmogorov-Arnold Network - Documentation

## 1. Executive Summary

This project implements a **Kolmogorov-Arnold Network (KAN)** for MNIST digit classification and provides a detailed proposal for extending it to a **Quantum KAN (QKAN)**.

### Key Results
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **97.17%** |
| Baseline Accuracy | 96.40% |
| Improvement | +0.77% |
| Model Parameters | 2,802,954 |
| Training Epochs | 15 |

---

## 2. Kolmogorov-Arnold Network Architecture

### 2.1 Theoretical Foundation

The **Kolmogorov-Arnold Representation Theorem** states that any continuous multivariate function can be represented as:

$$f(x_1, ..., x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

This means complex multivariate functions can be decomposed into compositions and sums of univariate functions.

### 2.2 KAN vs MLP

| Aspect | MLP | KAN |
|--------|-----|-----|
| **Activations** | Fixed (ReLU, etc.) | Learnable (basis functions) |
| **Weights** | Linear on nodes | On edges via basis coefficients |
| **Expressivity** | Width/depth | Basis function complexity |
| **Interpretability** | Low | Higher (visualizable functions) |

### 2.3 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (28×28)                       │
│                    Flattened to 784                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    KAN LAYER 1                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  For each input feature x_i:                        │    │
│  │  1. Compute Gaussian basis: φ_k(x) = exp(-(x-c_k)²/2σ²)│  │
│  │  2. Weighted sum: Σ w_ik · φ_k(x_i)                 │    │
│  │  3. Linear combination to output                    │    │
│  └─────────────────────────────────────────────────────┘    │
│  Input: 784 → Output: 256                                   │
│  Basis functions: 11 (grid_size + spline_order)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    SiLU ACTIVATION                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    KAN LAYER 2                               │
│  Input: 256 → Output: 128                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    SiLU ACTIVATION                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 LINEAR CLASSIFIER                            │
│                   128 → 10 classes                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 KAN Layer Implementation

```python
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_bases=10, sigma=0.3):
        # Fixed Gaussian centers in [-1, 1]
        self.centers = torch.linspace(-1, 1, num_bases)
        
        # Learnable coefficients: (in_features, num_bases, out_features)
        self.coefficients = nn.Parameter(...)
        
        # Base weight for residual connection
        self.base_weight = nn.Parameter(...)
    
    def forward(self, x):
        # 1. Compute Gaussian basis values
        basis = exp(-((x - centers)² / 2σ²))
        
        # 2. Apply learnable coefficients
        spline_out = einsum('bin,ino->bio', basis, coefficients)
        
        # 3. Sum over inputs + base transformation
        return spline_out.sum(dim=1) + x @ base_weight
```

### 2.5 Model Configuration

```python
config = {
    'model_type': 'efficient',
    'in_features': 784,
    'hidden_dims': [256, 128],
    'num_classes': 10,
    'grid_size': 8,
    'spline_order': 3,
    'basis_type': 'gaussian',
}
```

---

## 3. Training Details

### 3.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-5 |
| LR Scheduler | Cosine Annealing |
| Batch Size | 128 |
| Epochs | 15 |
| Gradient Clipping | max_norm=1.0 |

### 3.2 Data Preprocessing

- **Normalization**: Rescaled to [-1, 1] (symmetric range)
- **Train/Val Split**: 90%/10%

### 3.3 Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.4893 | 84.93% | 0.2845 | 91.50% |
| 5 | 0.1080 | 96.61% | 0.1568 | 95.37% |
| 10 | 0.0335 | 99.10% | 0.1272 | 96.58% |
| 15 | 0.0128 | 99.83% | 0.1227 | 96.78% |

---

## 4. Evaluation Results

### 4.1 Test Performance

- **Test Accuracy**: 97.17%
- **Best Validation Accuracy**: 96.78%

### 4.2 Baseline Comparison

| Model | Test Accuracy | Parameters |
|-------|---------------|------------|
| Task 9.ipynb (baseline) | 96.40% | ~2M |
| **Our Enhanced KAN** | **97.17%** | 2.8M |

---

## 5. Quantum KAN Proposal

### 5.1 Motivation

Quantum computing offers unique advantages for KAN:
1. **Quantum Basis Functions**: Quantum states can represent complex basis expansions
2. **Exponential State Space**: n qubits encode 2^n basis coefficients
3. **Quantum Interference**: Natural mechanism for combining basis functions

### 5.2 Proposed QKAN Architecture

#### Hybrid Classical-Quantum KAN

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT (784 features)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              CLASSICAL DIMENSIONALITY REDUCTION              │
│                   Linear (784 → n_qubits)                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    QUANTUM KAN LAYER                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  ANGLE ENCODING (analogous to placing on grid)      │    │
│  │  RY(x_i · π) for each qubit i                       │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  VARIATIONAL "BASIS" LAYER 1:                       │    │
│  │  - RX(θ), RY(θ), RZ(θ) rotations (learnable)        │    │
│  │  - CNOT entanglement (feature interaction)          │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  VARIATIONAL "BASIS" LAYER 2: (repeat)              │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  MEASUREMENT: ⟨Z_i⟩ for each qubit                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              CLASSICAL CLASSIFICATION HEAD                   │
│                   Linear (n_qubits → 10)                     │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 QKAN Implementation Sketch (PennyLane)

```python
import pennylane as qml
import torch.nn as nn

n_qubits = 8
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_kan_layer(inputs, weights):
    """
    Quantum KAN layer - learns basis-like functions.
    
    The variational circuit acts as a learnable function
    approximator, similar to how spline coefficients
    define the shape of classical basis functions.
    """
    # Data encoding (place input on "quantum grid")
    for i in range(n_qubits):
        qml.RY(inputs[i] * torch.pi, wires=i)
    
    # Variational layers (learn "basis shape")
    for layer in range(n_layers):
        # Parameterized rotations (basis coefficients)
        for i in range(n_qubits):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)
        
        # Entanglement (feature correlations)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])  # Circular
    
    # Measurements (basis function outputs)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumKANLayer(nn.Module):
    def __init__(self, in_features, out_features, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical pre-processing
        self.pre_net = nn.Linear(in_features, n_qubits)
        
        # Quantum weights (analogous to spline coefficients)
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )
        
        # Classical post-processing
        self.post_net = nn.Linear(n_qubits, out_features)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reduce dimensionality
        x = torch.tanh(self.pre_net(x))  # [-1, 1]
        
        # Quantum circuit
        outputs = []
        for i in range(batch_size):
            out = quantum_kan_layer(x[i], self.q_weights)
            outputs.append(torch.stack(out))
        
        x = torch.stack(outputs)
        return self.post_net(x)


class QuantumKAN(nn.Module):
    """Complete Quantum KAN for MNIST."""
    
    def __init__(self, in_features=784, n_qubits=8, n_layers=3, num_classes=10):
        super().__init__()
        self.qkan_layer = QuantumKANLayer(in_features, 64, n_qubits, n_layers)
        self.activation = nn.SiLU()
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.qkan_layer(x)
        x = self.activation(x)
        return self.classifier(x)
```

### 5.4 QKAN vs Classical KAN

| Aspect | Classical KAN | Quantum KAN |
|--------|---------------|-------------|
| Basis Functions | Gaussian/B-splines | Quantum feature maps |
| Parameters | O(in × out × grid) | O(layers × qubits × 3) |
| Expressivity | Polynomial | Potentially exponential |
| Computation | Classical | Quantum hardware/simulation |
| Scalability | Good | Limited by qubits |

### 5.5 Research Directions

1. **Quantum Basis Equivalence**: Can quantum circuits replicate B-spline behavior?
2. **Quantum Advantage**: Identify tasks where QKAN outperforms classical KAN
3. **Noise Resilience**: Develop error-robust QKAN architectures
4. **Efficient Encoding**: Explore amplitude encoding for dense inputs

---

## 6. Project Structure

```
Task 9/
├── src/
│   ├── __init__.py          # Package exports
│   ├── model.py             # KAN layers and networks
│   ├── dataset.py           # MNIST data loading
│   ├── training.py          # Training pipeline
│   └── utils.py             # Utilities
├── results/
│   ├── metrics.txt          # Performance metrics
│   ├── training_curves.png  # Training plots
│   ├── confusion_matrix.png # Confusion matrix
│   ├── classification_report.txt
│   └── kan_mnist.pt         # Saved model
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── PLANNING.md              # Planning document
├── DOCUMENTATION.md         # This file
└── README.md                # Quick start guide
```

---

## 7. Usage

### Training
```bash
# Basic training (matches baseline config)
python main.py --epochs 15 --save-model

# Enhanced multi-layer KAN
python main.py --enhanced --epochs 15 --save-model
```

### Inference
```python
from src.model import create_kan

config = {'model_type': 'efficient', 'hidden_dims': [256, 128], ...}
model = create_kan(config)
checkpoint = torch.load('results/kan_mnist.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

prediction = model(image).argmax(dim=1)
```

---

## 8. References

1. Liu, Z., et al. (2024). "KAN: Kolmogorov-Arnold Networks." arXiv:2404.19756
2. Kolmogorov, A. N. (1957). "On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition."
3. Arnold, V. I. (1957). "On functions of three variables."
4. de Boor, C. (1978). "A Practical Guide to Splines." Springer.

---

*Document Version: 1.0*
*Last Updated: Implementation Complete*
