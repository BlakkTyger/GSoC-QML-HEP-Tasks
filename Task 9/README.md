# Task IX: Kolmogorov-Arnold Network (KAN) for MNIST Classification

This task implements a **Kolmogorov-Arnold Network** using Gaussian basis functions for MNIST digit classification, achieving **97.17% test accuracy** — an improvement of +0.77% over the provided baseline. The documentation also includes a detailed proposal for extending the architecture to a **Quantum KAN (QKAN)**.

---

## Problem Statement

> *Implement a classical Kolmogorov-Arnold Network using basis-splines or some other KAN architecture and apply it to MNIST. Show its performance on the test data. Comment on potential ideas to extend this classical KAN architecture to a quantum KAN and sketch out the architecture in detail.*

---

## What is a KAN?

The **Kolmogorov-Arnold Representation Theorem** (1957) states that any continuous multivariate function can be represented as:

$$f(x_1, ..., x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

In other words, complex multivariate functions decompose into sums and compositions of **univariate functions**. KANs put this into practice: instead of fixed activations (ReLU, GELU) on nodes, KANs learn **activations on edges** using parameterized basis functions.

| Aspect | MLP | KAN |
|---|---|---|
| Activations | Fixed (ReLU, etc.) | Learnable (basis functions) |
| Weights | On nodes (linear transforms) | On edges (basis coefficients) |
| Expressivity | Depends on width/depth | Depends on basis complexity |
| Interpretability | Low | Higher (visualizable edge functions) |

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: PyTorch ≥ 2.0, torchvision, NumPy, Matplotlib, scikit-learn.

### Training

```bash
# Enhanced multi-layer KAN (best performance)
python main.py --enhanced --epochs 15 --save-model

# Baseline-matching KAN
python main.py --epochs 15 --save-model
```

### Inference

```python
import torch
from src.model import create_kan

model = create_kan({'model_type': 'efficient', 'hidden_dims': [256, 128]})
checkpoint = torch.load('results/kan_mnist.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

prediction = model(image).argmax(dim=1)
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `15` | Number of training epochs |
| `--enhanced` | `False` | Use enhanced multi-layer KAN |
| `--simple` | `False` | Use simple data loading (no val split) |
| `--save-model` | `False` | Save model checkpoint to `results/` |

---

## Project Structure

```
Task 9/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── main.py                # Entry point: argument parsing + pipeline orchestration
│
├── src/
│   ├── __init__.py
│   ├── model.py           # KANLayer (Gaussian basis), KAN network, model factory
│   ├── dataset.py         # MNIST data loading, normalization to [-1,1], train/val split
│   ├── training.py        # Training loop, validation, evaluation, gradient clipping
│   └── utils.py           # Plotting utilities (training curves, confusion matrix)
│
├── data/MNIST/            # Auto-downloaded dataset
│
└── results/
    ├── kan_mnist.pt              # Saved model checkpoint
    ├── metrics.txt               # Final performance metrics
    ├── classification_report.txt # Per-digit precision/recall/F1
    ├── training_curves.png       # Loss + accuracy over epochs
    └── confusion_matrix.png      # 10×10 confusion matrix heatmap
```

### Key Files

| File | Role |
|---|---|
| `model.py` | Implements `KANLayer` using Gaussian basis functions: for each input feature, computes 11 Gaussian radial basis values at fixed centers in [-1,1], multiplies by learnable coefficients, and sums across inputs. Includes a residual base weight for skip connections. The full `KAN` network stacks two KAN layers (784→256→128) with SiLU activation, followed by a linear classifier (128→10). |
| `dataset.py` | Loads MNIST, normalizes pixel values to [-1,1] (symmetric range for Gaussian centers), and creates a 90/10 train/val split. |
| `training.py` | Handles training with AdamW, cosine annealing LR, gradient clipping (max_norm=1.0), per-epoch validation, and test evaluation with classification report generation. |
| `utils.py` | Generates training curve plots and confusion matrix heatmaps. |

---

## Architecture

```
Input Image (28×28) → Flattened to 784
        ↓
┌─────── KAN Layer 1 ───────┐
│  784 input features         │
│  × 11 Gaussian basis funcs  │
│  = 784 × 11 = 8,624 basis  │
│  → Learnable coefficients   │
│  → Sum across inputs → 256  │
│  + Base weight residual     │
└────────────────────────────┘
        ↓  SiLU activation
┌─────── KAN Layer 2 ───────┐
│  256 → 128                  │
│  (same structure)           │
└────────────────────────────┘
        ↓  SiLU activation
Linear Classifier: 128 → 10
```

### Gaussian Basis Function

Each KAN layer replaces the standard linear+activation pattern with learnable univariate functions. For each input value `x`:

```
φ_k(x) = exp(−(x − c_k)² / 2σ²)
```

where `c_k` are 11 fixed centers equally spaced in [-1, 1] and `σ = 0.3`. The output for each edge is a weighted sum of these basis values — effectively learning a smooth univariate function per (input, output) pair.

### Model Configuration

| Parameter | Value |
|---|---|
| Input dimension | 784 (flattened 28×28) |
| Hidden layers | [256, 128] |
| Grid size | 8 |
| Spline order | 3 |
| Basis functions per edge | 11 (grid_size + spline_order) |
| Basis type | Gaussian (σ = 0.3) |
| Activation between layers | SiLU |
| **Total parameters** | **2,802,954** |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (weight decay = 1e-5) |
| Learning rate | 3e-4 |
| LR scheduler | Cosine annealing |
| Batch size | 128 |
| Epochs | 15 |
| Gradient clipping | max_norm = 1.0 |

---

## Results

### Test Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **97.17%** |
| Best Validation Accuracy | 96.78% |
| Final Training Accuracy | 99.83% |

### Baseline Comparison

| Model | Test Accuracy | Parameters |
|---|---|---|
| Task 9.ipynb (baseline) | 96.40% | ~2M |
| **Our Enhanced KAN** | **97.17%** | 2.8M |
| Improvement | **+0.77%** | — |

### Training Curves

![Training loss and accuracy curves for the KAN model](results/training_curves.png)

### Confusion Matrix

![10×10 confusion matrix showing per-digit classification performance](results/confusion_matrix.png)

### Per-Digit Performance

| Digit | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 97.4% | 98.7% | 98.0% |
| 1 | 98.7% | 98.9% | 98.8% |
| 2 | 97.1% | 96.9% | 97.0% |
| 3 | 96.8% | 97.3% | 97.0% |
| 4 | 98.0% | 96.7% | 97.4% |
| 5 | 97.5% | 96.6% | 97.1% |
| 6 | 97.5% | 97.2% | 97.3% |
| 7 | 96.9% | 96.7% | 96.8% |
| 8 | 96.3% | 96.5% | 96.4% |
| 9 | 95.5% | 95.9% | 95.7% |

---

## Quantum KAN Proposal

The KAN's edge-based learnable functions map naturally to quantum circuits — variational quantum circuits learn smooth functions in Hilbert space, analogous to how Gaussian basis functions learn smooth functions in classical space.

### Proposed QKAN Architecture

```
Input (784) → Classical Linear (784 → n_qubits) → tanh → [-1, 1]
        ↓
┌──────── Quantum KAN Layer ────────┐
│  Angle Encoding: RY(x_i · π)      │
│  Variational "Basis" Layer 1:      │
│    RX(θ), RY(θ), RZ(θ) per qubit  │
│    + Circular CNOT entanglement    │
│  Variational "Basis" Layer 2       │
│  ⟨Z_i⟩ measurements               │
└────────────────────────────────────┘
        ↓
Classical Linear (n_qubits → 10) → Classification
```

The variational layers act as **quantum basis functions** — each layer's rotations + entanglement define a smooth, learnable function in the exponentially large Hilbert space, analogous to how Gaussian basis coefficients define a smooth function in ℝ.

### QKAN vs Classical KAN

| Aspect | Classical KAN | Quantum KAN |
|---|---|---|
| Basis functions | Gaussian / B-splines | Quantum feature maps |
| Parameters | O(in × out × grid) | O(layers × qubits × 3) |
| Expressivity | Polynomial | Potentially exponential |
| Computation | Classical | Quantum hardware/simulation |

---

## References

1. Z. Liu et al., *"KAN: Kolmogorov-Arnold Networks"*, [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
2. A. N. Kolmogorov, *"On the Representation of Continuous Functions of Many Variables by Superposition of Continuous Functions of One Variable and Addition"* (1957).
3. V. I. Arnold, *"On Functions of Three Variables"* (1957).
4. C. de Boor, *"A Practical Guide to Splines"*, Springer (1978).
