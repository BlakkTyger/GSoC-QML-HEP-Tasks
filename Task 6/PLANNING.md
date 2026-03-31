# Task VI: Quantum Representation Learning - Planning Document

## Table of Contents
1. [Task Overview](#1-task-overview)
2. [Dataset Understanding](#2-dataset-understanding)
3. [Theoretical Foundations](#3-theoretical-foundations)
4. [Architecture Design](#4-architecture-design)
5. [Implementation Plan](#5-implementation-plan)
6. [References](#6-references)

---

## 1. Task Overview

### 1.1 Objective
Implement a quantum representation learning scheme using contrastive loss to learn meaningful quantum representations of MNIST images through:
- Parameterized quantum state preparation with trainable encoding
- SWAP test for quantum fidelity measurement
- Contrastive learning optimization across all 10 digit classes

### 1.2 Core Requirements
1. **Load MNIST dataset** - Full 28×28 grayscale images, all 10 classes
2. **Quantum state preparation function** - Trainable parameters for encoding images
3. **Dual-image circuit with SWAP test** - Embed two images and measure fidelity
4. **Contrastive loss training** - Maximize fidelity for same-class, minimize for different-class pairs

### 1.3 Success Criteria
- Achieve >85% accuracy on same/different class pair classification
- Clear fidelity separation between same-class and different-class pairs
- Model generalizes to unseen test images

---

## 2. Dataset Understanding

### 2.1 MNIST Dataset Characteristics

| Property | Value |
|----------|-------|
| Image dimensions | 28 × 28 pixels |
| Color depth | 8-bit grayscale (0-255) |
| Total features | 784 pixels |
| Number of classes | 10 (digits 0-9) |
| Training samples | 60,000 |
| Test samples | 10,000 |

### 2.2 Preprocessing Strategy

**Quadrant Mean Pooling:**
Rather than resizing the image, we compute the mean intensity of each 14×14 quadrant:

```python
def preprocess_image(img):
    q1 = np.mean(img[:14, :14])   # Top-left
    q2 = np.mean(img[:14, 14:])   # Top-right  
    q3 = np.mean(img[14:, :14])   # Bottom-left
    q4 = np.mean(img[14:, 14:])   # Bottom-right
    return [q1, q2, q3, q4]
```

**Rationale:**
- Preserves spatial structure (digit position information)
- Reduces 784 features to 4 meaningful aggregates
- More robust than pixel-level features
- Natural mapping to 4 qubits

---

## 3. Theoretical Foundations

### 3.1 SWAP Test for Fidelity Measurement

**Definition:** The SWAP test measures the overlap (fidelity) between two quantum states |ψ⟩ and |φ⟩.

**Circuit Structure:**
```
|0⟩_anc ─[H]───●────●────●────●───[H]─[M]
               │    │    │    │
|ψ⟩     ──────[×]──[×]──[×]──[×]─────────
               │    │    │    │
|φ⟩     ──────[×]──[×]──[×]──[×]─────────
```

**Measurement Result:**
- P(ancilla = 0) = (1 + |⟨ψ|φ⟩|²) / 2
- Expectation ⟨Z⟩ = |⟨ψ|φ⟩|² = Fidelity

### 3.2 Trainable Quantum Encoding

**Key Innovation:** Instead of fixed angle encoding, use trainable linear transformation:

```
θ = params[i, 0] * x[i] + params[i, 1]
```

Where:
- `x[i]` is the i-th feature (quadrant mean)
- `params[i, 0]` is the learnable scale
- `params[i, 1]` is the learnable bias

This allows the network to learn optimal rotation angles for each feature.

### 3.3 Separate Encoders for Two Images

**Design Choice:** Use different rotation gates for each image:
- **Image 1:** RY rotations (Y-axis)
- **Image 2:** RX rotations (X-axis)

**Rationale:**
- Creates distinct encoding pathways
- Prevents trivial solutions where both encodings collapse
- Allows model to learn complementary representations

### 3.4 Contrastive Loss Function

**Formulation:**
```python
loss = label * (1 - fidelity)² + (1 - label) * fidelity²
```

Where:
- `label = 1` for same-class pairs → minimize (1 - fidelity)²
- `label = 0` for different-class pairs → minimize fidelity²

**Properties:**
- No margin parameter needed
- Symmetric treatment of positive/negative pairs
- Smooth gradients for optimization

---

## 4. Architecture Design

### 4.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 QUANTUM SIMILARITY NETWORK                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MNIST Image 1 (28×28)    MNIST Image 2 (28×28)                 │
│         │                        │                               │
│         ▼                        ▼                               │
│  ┌─────────────┐          ┌─────────────┐                       │
│  │  Quadrant   │          │  Quadrant   │                       │
│  │  Pooling    │          │  Pooling    │                       │
│  └──────┬──────┘          └──────┬──────┘                       │
│         │ [4 features]           │ [4 features]                  │
│         ▼                        ▼                               │
│  ┌─────────────┐          ┌─────────────┐                       │
│  │ RY Encoding │          │ RX Encoding │                       │
│  │ (params1)   │          │ (params2)   │                       │
│  └──────┬──────┘          └──────┬──────┘                       │
│         │                        │                               │
│         └──────────┬─────────────┘                               │
│                    ▼                                             │
│           ┌───────────────┐                                      │
│           │   SWAP TEST   │                                      │
│           │  (9 qubits)   │                                      │
│           └───────┬───────┘                                      │
│                   │                                              │
│                   ▼                                              │
│              Fidelity                                            │
│                   │                                              │
│                   ▼                                              │
│         ┌─────────────────┐                                      │
│         │ Contrastive Loss│                                      │
│         └─────────────────┘                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Circuit Specifications

| Component | Specification |
|-----------|---------------|
| Total qubits | 9 (1 ancilla + 4 + 4) |
| Encoding (Image 1) | 4× RY gates with trainable params |
| Encoding (Image 2) | 4× RX gates with trainable params |
| SWAP test | 4× CSWAP gates |
| Trainable parameters | 16 (4 qubits × 2 params × 2 images) |

### 4.3 Parameter Structure

```python
params1 = [[scale_0, bias_0],   # Qubit 0, Image 1
           [scale_1, bias_1],   # Qubit 1, Image 1
           [scale_2, bias_2],   # Qubit 2, Image 1
           [scale_3, bias_3]]   # Qubit 3, Image 1

params2 = [[scale_0, bias_0],   # Qubit 0, Image 2
           [scale_1, bias_1],   # Qubit 1, Image 2
           [scale_2, bias_2],   # Qubit 2, Image 2
           [scale_3, bias_3]]   # Qubit 3, Image 2
```

---

## 5. Implementation Plan

### 5.1 Project Structure

```
Task 6/
├── main.py              # Complete implementation
├── results/
│   ├── training_results.png
│   ├── metrics.txt
│   └── model.pt
├── data/                # MNIST data (auto-downloaded)
├── PLANNING.md          # This document
├── DOCUMENTATION.md     # Technical documentation
├── README.md            # Task description
└── requirements.txt     # Dependencies
```

### 5.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Training samples | 4000 |
| Epochs | 50 |
| Iterations per epoch | 100 |
| Optimizer | Adam |
| Learning rate | 0.02 |
| Batch sampling | Random pairs |
| Classes | All 10 MNIST digits |

### 5.3 Evaluation Protocol

1. Sample random pairs from test set
2. Compute fidelity using trained model
3. Classify: fidelity ≥ 0.5 → same class
4. Compute accuracy, fidelity statistics

---

## 6. References

### 6.1 Primary Sources

1. **SWAP Test** - Buhrman et al. "Quantum Fingerprinting"
   - Original formulation of SWAP test for state comparison
   - Wikipedia: https://en.wikipedia.org/wiki/Swap_test

2. **Contrastive Learning** - Chen et al. "SimCLR"
   - Classical foundation for contrastive representation learning
   - Applied to quantum domain with fidelity as similarity metric

3. **Quantum Machine Learning** - Schuld & Petruccione
   - Comprehensive reference for variational quantum circuits
   - PennyLane documentation: https://pennylane.ai/qml/

### 6.2 Implementation References

4. **PennyLane** - Quantum ML framework
   - PyTorch interface for hybrid quantum-classical training
   - Automatic differentiation through quantum circuits

5. **MNIST Dataset** - LeCun et al.
   - Standard benchmark for image classification
   - 10 handwritten digit classes

---

*Document Version: 2.0*  
*Last Updated: March 2025*
