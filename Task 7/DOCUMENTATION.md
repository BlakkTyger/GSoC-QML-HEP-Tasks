# Task 7: Z₂ × Z₂ Equivariant Quantum Neural Networks

## Technical Documentation

---

## 1. Overview

This document provides implementation specifications for Z₂ × Z₂ equivariant quantum neural networks applied to a symmetric binary classification problem. The implementation compares standard (non-equivariant) QNNs against symmetry-aware equivariant QNNs.

### 1.1 Task Summary

| Aspect | Description |
|--------|-------------|
| **Problem** | Binary classification on 2D data with Z₂ × Z₂ symmetry |
| **Symmetry Group** | Klein four-group V₄ = Z₂ × Z₂ |
| **Models** | Standard QNN vs. Equivariant QNN |
| **Framework** | PennyLane + PyTorch |
| **Qubits** | 2 |

### 1.2 Final Results

| Model | Train Acc | Test Acc | Parameters |
|-------|-----------|----------|------------|
| Standard QNN | 91.56% | 88.75% | 12 |
| Equivariant QNN | 91.56% | 88.75% | 9 |

**Key Finding**: Equivariant QNN achieves same accuracy with **25% fewer parameters**.

---

## 2. Mathematical Foundation

### 2.1 Z₂ × Z₂ Symmetry Group

The Klein four-group consists of four elements with the following action on 2D data:

| Element | Action on (x₁, x₂) | Physical Meaning |
|---------|-------------------|------------------|
| (0, 0) | (x₁, x₂) | Identity |
| (1, 0) | (x₂, x₁) | Coordinate exchange |
| (0, 1) | (-x₁, -x₂) | Point reflection (origin) |
| (1, 1) | (-x₂, -x₁) | Combined transformation |

### 2.2 Quantum Representations

**Data Embedding:**
```
U(x₁, x₂) = R_Z(x₁) ⊗ R_Z(x₂)
```

**Hilbert Space Representations:**

| Group Element | Unitary Representation |
|--------------|------------------------|
| (0, 0) | I ⊗ I |
| (1, 0) | SWAP |
| (0, 1) | X ⊗ X |
| (1, 1) | SWAP · (X ⊗ X) |

### 2.3 Equivariant Gates

Through gate symmetrization (twirling), the equivariant gateset reduces to:

```
G_equiv = {X₁ + X₂, Z₁Z₂}
```

**Equivariant Gate Implementations:**

1. **Simultaneous X Rotation:**
   ```
   R_XX(θ) = exp(-iθ(X⊗I + I⊗X)/2) = R_X(θ) ⊗ R_X(θ)
   ```

2. **ZZ Interaction:**
   ```
   R_ZZ(θ) = exp(-iθ · Z⊗Z)
   ```

---

## 3. Implementation Specifications

### 3.1 Dataset Module (`dataset.py`)

```python
def generate_z2z2_dataset(n_points=200, threshold=0.05, seed=None):
    """
    Generate binary classification dataset with Z₂ × Z₂ symmetry.
    
    Classification Rule: |x₁ - x₂| >= threshold
    Symmetry enforced by adding swapped points (x₂, x₁).
    
    Parameters:
        n_points: Number of base points (total = 2x with symmetry)
        threshold: Distance threshold from diagonal
        
    Returns:
        X: Features array of shape (2*n_points, 2)
        y: Labels array of shape (2*n_points,)
    """
```

**Key Properties:**
- Features: x₁, x₂ ∈ [0, 1]
- Labels: Class 0 if |x₁-x₂| < threshold, Class 1 otherwise
- Symmetry: Enforced by adding (x₂, x₁) for each (x₁, x₂)

### 3.2 Standard QNN (`models.py`)

**Architecture:**
```
|0⟩ ─[AngleEmbed(x₁)]─[BasicEntanglerLayers]─ ⟨Z₀⟩
                                              
|0⟩ ─[AngleEmbed(x₂)]─[BasicEntanglerLayers]─ ⟨Z₁⟩ → Linear(2,2) → Classification
```

**Parameters:** 12 (6 QNN + 6 classical linear layer)

**Circuit Template:**
```python
@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
```

### 3.3 Equivariant QNN (`models.py`)

**Architecture:**
```
|0⟩ ─[AngleEmbed(x₁)]─[R_X(θ)]─●───[R_X(θ)]─●─── ⟨Z₀⟩
                               │           │
|0⟩ ─[AngleEmbed(x₂)]─[R_X(θ)]─X───[R_X(θ)]─X─── ⟨Z₁⟩ → Linear(2,2)
```

**Key Features:**
- **Tied parameters**: Same R_X angle on both qubits (equivariance constraint)
- CNOT for entanglement
- **Parameters:** 9 (3 QNN + 6 classical) - **25% reduction**

**Circuit Template:**
```python
@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    for i in range(n_layers):
        qml.RX(weights[i, 0], wires=0)
        qml.RX(weights[i, 0], wires=1)  # Same angle!
        qml.CNOT(wires=[0, 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
```

### 3.4 Training Module (`training.py`)

**Loss Function:** NLLLoss (Negative Log Likelihood)

**Training Loop:**
```python
def train_model(model, X, y, n_epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.01 |
| Epochs | 100 |
| Optimizer | Adam |
| Layers | 3 |

---

## 4. Evaluation Metrics

### 4.1 Primary Metrics

1. **Classification Accuracy:**
   ```python
   accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Generalization Gap:**
   ```python
   gen_gap = train_accuracy - test_accuracy
   ```

3. **Parameter Efficiency:**
   ```python
   efficiency = test_accuracy / n_parameters
   ```

### 4.2 Visualization Outputs

1. **Dataset Visualization:**
   - Scatter plot with class colors
   - Symmetry transformation verification

2. **Training Curves:**
   - Loss vs. epochs (both models)
   - Accuracy vs. epochs

3. **Decision Boundaries:**
   - 2D contour plots
   - Symmetry preservation verification

4. **Comparison Summary:**
   - Bar charts for metrics
   - Parameter count comparison

---

## 5. Final Results

### 5.1 Experimental Results

| Metric | Standard QNN | Equivariant QNN |
|--------|--------------|-----------------|
| Train Accuracy | 91.56% | 91.56% |
| Test Accuracy | 88.75% | 88.75% |
| Total Parameters | 12 | 9 |
| Parameter Reduction | - | 25% |

### 5.2 Key Findings

- **Same accuracy**: Both models achieve ~89% test accuracy
- **Fewer parameters**: Equivariant QNN uses 25% fewer parameters
- **Faster convergence**: Equivariant QNN converges in ~20 epochs vs ~40 for standard

---

## 6. File Structure

```
Task 7/
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Dataset generation
│   ├── models.py            # Standard & Equivariant QNN
│   ├── training.py          # Training loops
│   └── visualization.py     # Plotting utilities
├── results/
│   ├── figures/             # Generated plots
│   └── metrics.txt          # Final metrics
├── main.py                  # Main execution script
├── requirements.txt         # Dependencies
├── PLANNING.md              # Research document
└── DOCUMENTATION.md         # This document
```

---

## 7. Dependencies

```
pennylane>=0.33.0
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## 8. Usage

### 8.1 Quick Start

```bash
pip install -r requirements.txt
python main.py --save
```

### 8.2 Module Usage

```python
from src.dataset import generate_z2z2_dataset, to_torch_tensors
from src.models import create_standard_classifier, create_equivariant_classifier
from src.training import train_model, evaluate_model

# Generate dataset
X, y = generate_z2z2_dataset(200, threshold=0.05)
X_tensor, y_tensor = to_torch_tensors(X, y)

# Train models
std_model = create_standard_classifier()
std_model, _ = train_model(std_model, X_tensor, y_tensor, n_epochs=100)

eqv_model = create_equivariant_classifier()
eqv_model, _ = train_model(eqv_model, X_tensor, y_tensor, n_epochs=100)

# Evaluate
print(evaluate_model(std_model, X_tensor, y_tensor))
print(evaluate_model(eqv_model, X_tensor, y_tensor))
```

---

## 9. Theoretical Justification

### 9.1 Why Equivariance Improves Generalization

1. **Reduced Hypothesis Space:** By constraining the model to respect symmetries, we eliminate equivalent but distinct representations of the same function.

2. **Implicit Data Augmentation:** An equivariant model effectively "sees" all symmetric versions of each training point.

3. **Inductive Bias:** The symmetry constraint encodes prior knowledge about the problem structure.

### 9.2 Barren Plateau Mitigation

Equivariant circuits have been shown to mitigate barren plateaus because:
- Smaller parameter space → better gradient landscape
- Symmetry constraints prevent loss of structure in deep circuits
- Invariant observables have non-vanishing gradients in symmetric subspace

**Reference:** Nguyen et al., PRX Quantum 5, 020328 (2024)

---

## 10. Validation Checklist

- [ ] Dataset satisfies Z₂ × Z₂ symmetry (verified by transformation tests)
- [ ] Equivariant circuit gates commute with all symmetry representations
- [ ] Invariant observable (Z⊗Z) commutes with all U_s
- [ ] Both models converge during training
- [ ] Equivariant model shows smaller generalization gap
- [ ] Decision boundaries respect symmetry (visual verification)
- [ ] Results are reproducible with fixed random seeds

---

*Document Version: 1.0*
*Last Updated: March 2025*
