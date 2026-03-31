# Task IX: Kolmogorov-Arnold Network - Planning Document

## 1. Task Overview

**Objective:** Implement a classical Kolmogorov-Arnold Network (KAN) for MNIST classification using B-splines or alternative basis functions, achieve competitive accuracy (≥96.40% baseline), and propose a detailed Quantum KAN (QKAN) architecture.

**Deliverables:**
1. Modular KAN implementation with training/evaluation pipeline
2. Test accuracy matching or exceeding baseline (96.40%)
3. Detailed QKAN architecture proposal with implementation sketch

---

## 2. Baseline Analysis

### 2.1 Task 9.ipynb (Gaussian-basis KAN)
| Parameter | Value |
|-----------|-------|
| Input Features | 784 (28×28 flattened) |
| Basis Functions | Gaussian (10 centers) |
| Hidden Dimension | 256 |
| Sigma | 0.3 |
| Learning Rate | 3e-4 |
| Epochs | 10 |
| **Test Accuracy** | **96.40%** |

**Architecture:**
```
Input (784) → KANLayer (Gaussian basis) → ReLU → Linear (256→10) → Output
```

**Key Observations:**
- Simple single KAN layer architecture
- Gaussian basis functions approximate spline behavior
- Centers uniformly distributed in [-1, 1]
- Learnable weights per input-feature and basis function

---

## 3. Kolmogorov-Arnold Representation Theorem

### 3.1 Mathematical Foundation

The **Kolmogorov-Arnold Representation Theorem** (1957) states that any continuous multivariate function can be represented as:

$$f(x_1, ..., x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

Where:
- $\phi_{q,p}: [0,1] \rightarrow \mathbb{R}$ are univariate inner functions
- $\Phi_q: \mathbb{R} \rightarrow \mathbb{R}$ are univariate outer functions
- Only univariate functions are needed to represent any multivariate function!

### 3.2 Implications for Neural Networks

**Traditional MLPs:**
- Fixed activation functions (ReLU, sigmoid, etc.)
- Learnable linear weights
- Approximation via depth and width

**KANs:**
- Learnable activation functions (on edges)
- No linear weights (activations do the work)
- Approximation via basis function complexity

### 3.3 KAN vs MLP Comparison

| Aspect | MLP | KAN |
|--------|-----|-----|
| Activations | Fixed (ReLU, etc.) | Learnable (splines) |
| Weights | On nodes | On edges (via basis) |
| Expressivity | Width/depth | Basis complexity |
| Interpretability | Low | Higher (spline visualization) |
| Parameters | Dense matrices | Basis coefficients |

---

## 4. Basis Function Research

### 4.1 B-Splines (Original KAN Paper)

**B-Spline Definition:**
B-splines of order k are piecewise polynomials defined recursively:

$$B_{i,0}(x) = \begin{cases} 1 & t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$B_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i,k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(x)$$

**Advantages:**
- Compact support (local influence)
- Smooth derivatives
- Partition of unity property
- Well-suited for function approximation

**Implementation:**
```python
def bspline_basis(x, grid, k=3):
    # Recursive B-spline computation
    # k=3 gives cubic splines (most common)
```

### 4.2 Gaussian Basis (Baseline)

$$\phi(x) = \exp\left(-\frac{(x - c)^2}{2\sigma^2}\right)$$

**Advantages:**
- Simple implementation
- Smooth and differentiable
- Good approximation properties

**Disadvantages:**
- Global support (less efficient)
- Not true splines

### 4.3 Chebyshev Polynomials (Alternative)

$$T_n(x) = \cos(n \cdot \arccos(x))$$

**Advantages:**
- Orthogonal basis
- Excellent approximation properties
- Used in ChebyKAN

### 4.4 Fourier Basis (Alternative)

$$\phi_k(x) = \sin(k\pi x), \cos(k\pi x)$$

**Advantages:**
- Orthogonal
- Good for periodic patterns

### 4.5 Chosen Approach: B-Spline KAN

We'll implement **B-spline basis** for authenticity to the original KAN paper, with Gaussian as fallback for simplicity.

---

## 5. Architecture Design

### 5.1 Single-Layer KAN (Baseline Match)

```
Input (784)
    ↓
KANLayer (B-spline basis, 10 bases, 256 hidden)
    ↓
ReLU
    ↓
Linear (256 → 10)
    ↓
Output (10 classes)
```

### 5.2 Multi-Layer KAN (Enhanced)

```
Input (784)
    ↓
KANLayer1 (784 → 128, spline_order=3, grid_size=5)
    ↓
SiLU activation
    ↓
KANLayer2 (128 → 64, spline_order=3, grid_size=5)
    ↓
SiLU activation
    ↓
Linear (64 → 10)
    ↓
Output (10 classes)
```

### 5.3 Efficient KAN Layer Design

```python
class EfficientKANLayer(nn.Module):
    """
    KAN Layer with B-spline basis functions.
    
    For each edge (i,j) from input i to output j:
    - Compute B-spline basis values for input[i]
    - Linearly combine with learnable coefficients
    - Sum contributions to output[j]
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        # Grid points for B-splines
        # Learnable coefficients: (in_features, out_features, grid_size + spline_order)
        # Base weight for residual connection (optional)
```

---

## 6. Quantum KAN Research

### 6.1 Quantum Computing Advantages for KAN

1. **Function Approximation**: Quantum circuits can approximate functions with exponential efficiency
2. **Basis Encoding**: Quantum states naturally encode basis function expansions
3. **Parallelism**: Superposition allows parallel evaluation of basis functions

### 6.2 Proposed Quantum KAN Architectures

#### Option A: Quantum Basis Functions (Recommended)

Replace classical basis functions with quantum feature maps:

```
Classical Input x
        ↓
Quantum Feature Map |ψ(x)⟩
        ↓
Variational Quantum Circuit (learnable "coefficients")
        ↓
Measurement → Classical Output
```

**Quantum Feature Map Examples:**
- Angle encoding: $|ψ(x)⟩ = \prod_i R_Y(x_i)|0⟩$
- Amplitude encoding: Encode basis coefficients in amplitudes
- IQP encoding: $e^{i x_i x_j Z_i Z_j}$ for feature interactions

#### Option B: Quantum Spline Approximation

Use quantum circuits to compute spline-like functions:

```
Input x encoded in rotation angles
        ↓
Layered quantum circuit (approximates spline shape)
        ↓
Measurement gives spline value
```

#### Option C: Hybrid Classical-Quantum KAN

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
│  │  Angle Encoding: RY(x_i) for each qubit             │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Quantum "Basis" Layer 1:                           │    │
│  │  - Parameterized rotations (learnable "splines")    │    │
│  │  - Entangling gates (CNOT ladder)                   │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Quantum "Basis" Layer 2: (repeat)                  │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Measurement: ⟨Z_i⟩ for each qubit                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              CLASSICAL CLASSIFICATION HEAD                   │
│                   Linear (n_qubits → 10)                     │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 QKAN Implementation Sketch (PennyLane)

```python
import pennylane as qml
import torch.nn as nn

n_qubits = 8
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_kan_layer(inputs, weights):
    """
    Quantum KAN layer that learns basis-like functions.
    
    inputs: (n_qubits,) - encoded features
    weights: (n_layers, n_qubits, 3) - rotation angles (RX, RY, RZ)
    """
    # Data encoding (analogous to placing input on spline grid)
    for i in range(n_qubits):
        qml.RY(inputs[i] * np.pi, wires=i)
    
    # Variational layers (learn the "spline coefficients")
    for layer in range(n_layers):
        # Single-qubit rotations (learnable basis coefficients)
        for i in range(n_qubits):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)
        
        # Entanglement (captures correlations between features)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])  # Circular
    
    # Measurements (output of "learned basis functions")
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumKANLayer(nn.Module):
    def __init__(self, in_features, out_features, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical pre-processing (dimensionality reduction)
        self.pre_net = nn.Linear(in_features, n_qubits)
        
        # Quantum circuit weights (learnable "basis coefficients")
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )
        
        # Classical post-processing
        self.post_net = nn.Linear(n_qubits, out_features)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reduce to qubit count
        x = torch.tanh(self.pre_net(x))  # [-1, 1] range
        
        # Apply quantum circuit to each sample
        outputs = []
        for i in range(batch_size):
            out = quantum_kan_layer(x[i], self.q_weights)
            outputs.append(torch.stack(out))
        
        x = torch.stack(outputs)
        return self.post_net(x)
```

### 6.4 QKAN vs Classical KAN

| Aspect | Classical KAN | Quantum KAN |
|--------|---------------|-------------|
| Basis Functions | B-splines, Gaussian | Quantum feature maps |
| Parameters | O(in × out × grid) | O(layers × qubits × 3) |
| Expressivity | Polynomial | Potentially exponential |
| Computation | Classical | Quantum simulation/hardware |
| Scalability | Linear | Limited by qubits |

### 6.5 Research Directions

1. **Quantum Advantage**: Can QKAN learn functions more efficiently than classical KAN?
2. **Basis Equivalence**: What quantum circuits approximate B-spline behavior?
3. **Noise Resilience**: How robust is QKAN to quantum noise?
4. **Hybrid Training**: Optimal classical-quantum layer combinations

---

## 7. Implementation Plan

### 7.1 File Structure

```
Task 9/
├── src/
│   ├── __init__.py
│   ├── model.py          # KAN layers and networks
│   ├── dataset.py        # MNIST data loading
│   ├── training.py       # Training pipeline
│   └── utils.py          # Utilities
├── results/
│   ├── metrics.txt
│   ├── training_curves.png
│   └── confusion_matrix.png
├── main.py
├── requirements.txt
├── PLANNING.md
├── DOCUMENTATION.md
└── README.md
```

### 7.2 Model Configurations

**Config 1: Gaussian KAN (Baseline Match)**
```python
config = {
    'basis_type': 'gaussian',
    'num_bases': 10,
    'hidden_dim': 256,
    'sigma': 0.3
}
```

**Config 2: B-Spline KAN (Enhanced)**
```python
config = {
    'basis_type': 'bspline',
    'grid_size': 5,
    'spline_order': 3,
    'hidden_dims': [128, 64]
}
```

### 7.3 Training Configuration

```python
training_config = {
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'epochs': 15,
    'batch_size': 128,
    'scheduler': 'cosine'
}
```

---

## 8. Risk Assessment

| Risk | Mitigation |
|------|------------|
| B-spline numerical instability | Use efficient recursive computation, clamp values |
| Overfitting | Regularization, dropout, grid size tuning |
| Slow training | Efficient basis computation, batch processing |
| Lower accuracy than baseline | Hyperparameter tuning, deeper architecture |

---

## 9. References

1. Liu, Z., et al. (2024). "KAN: Kolmogorov-Arnold Networks." arXiv:2404.19756
2. Kolmogorov, A. N. (1957). "On the representation of continuous functions..."
3. Arnold, V. I. (1957). "On functions of three variables."
4. de Boor, C. (1978). "A Practical Guide to Splines."

---

*Document Version: 1.0*
*Last Updated: During Implementation*
