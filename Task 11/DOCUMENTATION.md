# Task 11: PQC Embedding with MLP Parameter Estimation

## Overview

This task implements a hybrid quantum-classical model for function approximation using:
- **MLP (Classical)**: Maps scalar inputs to PQC parameters
- **PQC (Quantum)**: Generates quantum states from learned parameters
- **Output Scaling**: Maps expectation values to target range

The goal is to learn mappings from normally distributed inputs to multi-dimensional targets.

---

## Architecture

### Model Components

```
Input x ~ N(0,1) → MLP → PQC Parameters → Quantum Circuit → Scaled Output
     (1)        (32→64→15)    (3×5)           (5 qubits)        (4)
```

### 1. MLP Network
- **Input**: Scalar from normal distribution (dim=1)
- **Hidden Layers**: 32 → 64 neurons with ReLU
- **Output**: 15 PQC parameters (3 layers × 5 qubits)

### 2. Parameterized Quantum Circuit
- **Qubits**: 5
- **Layers**: 3
- **Structure per layer**:
  - RY rotations on all qubits (parameterized)
  - Ring CNOT entanglement: CNOT(i, (i+1) mod 5)
- **Measurement**: Pauli-Z expectations on first 4 qubits

### 3. Output Scaling
- Learnable scale and bias parameters
- Maps [-1, 1] expectation values to target range

---

## Target Function

For input scalar `x`, the target is:
```
y = [x, sin(x), cos(x), x²]
```

This creates a 4-dimensional regression problem from 1D input.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Samples | 2048 |
| Train/Test Split | 80/20 |
| Batch Size | 32 |
| Epochs | 50 |
| Initial Learning Rate | 0.005 |
| LR Scheduler | StepLR (step=20, γ=0.5) |
| Loss Function | MSE |
| Optimizer | Adam |

---

## File Structure

```
Task 11/
├── src/
│   ├── __init__.py      # Package exports
│   ├── model.py         # HybridModel and PQC definition
│   ├── dataset.py       # Data generation utilities
│   └── training.py      # Training loop and evaluation
├── results/
│   ├── model.pt         # Saved model checkpoint
│   ├── metrics.txt      # Evaluation metrics
│   └── training_curve.png
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── PLANNING.md          # Design decisions
├── DOCUMENTATION.md     # This file
└── README.md            # Quick start guide
```

---

## Usage

### Training
```bash
python main.py --epochs 50 --lr 0.005
```

### Command Line Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--num_samples` | 2048 | Number of data samples |
| `--batch_size` | 32 | Training batch size |
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 0.005 | Learning rate |
| `--scheduler_step` | 20 | LR decay step |
| `--scheduler_gamma` | 0.5 | LR decay factor |
| `--device` | cpu | Device (cpu/cuda) |
| `--seed` | 42 | Random seed |
| `--results_dir` | results | Output directory |

---

## Key Design Decisions

### 1. Output Scaling Layer
**Problem**: PQC outputs are bounded in [-1, 1] (Pauli-Z expectations), but targets like `x` and `x²` can exceed this range.

**Solution**: Added learnable `scale` and `bias` parameters:
```python
output = scale * pqc_output + bias
```

### 2. Backprop Differentiation
Used `diff_method='backprop'` instead of `'parameter-shift'` for faster gradient computation during training.

### 3. 5 Qubits with 3 Layers
- Increased from baseline (4 qubits, 2 layers)
- Better expressibility for complex function approximation
- Manageable simulation cost

### 4. Wider MLP
- Expanded from [16, 16] to [32, 64]
- Better capacity to learn input-to-parameter mapping

---

## Results

### Performance Summary

| Metric | Baseline | Our Model | Improvement |
|--------|----------|-----------|-------------|
| Test MSE | 0.5397 | **0.0541** | **90.0%** |

### Per-Target MSE

| Target | MSE | Notes |
|--------|-----|-------|
| x | 0.0086 | Excellent fit |
| sin(x) | 0.0023 | Best performance |
| cos(x) | 0.0026 | Excellent fit |
| x² | 0.2229 | Hardest (unbounded range) |

### Analysis

The model achieves a **90% improvement** over the baseline. Key observations:

1. **Trigonometric functions** (sin, cos) are learned best - naturally bounded to [-1, 1]
2. **Linear function** (x) is well-approximated with the output scaling layer
3. **Quadratic function** (x²) is hardest due to unbounded positive range

---

## Comparison with Baseline

| Aspect | Baseline | Our Implementation |
|--------|----------|-------------------|
| Qubits | 4 | 5 |
| PQC Layers | 2 | 3 |
| MLP Architecture | 1→16→16→8 | 1→32→64→15 |
| Output Scaling | None | Learnable |
| Samples | 1024 | 1024 |
| Epochs | 20 | 30 |
| LR Schedule | None | StepLR |
| Train/Test Split | No | Yes (80/20) |
| **Test MSE** | 0.5397 | **0.0541** |

---

## Dependencies

- PyTorch >= 2.0.0
- PennyLane >= 0.33.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0

---

## References

1. Benedetti et al., "Parameterized quantum circuits as machine learning models" (2019)
2. Schuld & Petruccione, "Machine Learning with Quantum Computers" (2021)
3. PennyLane Documentation: https://pennylane.ai/
