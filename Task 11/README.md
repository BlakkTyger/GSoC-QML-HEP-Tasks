# Task XI: PQC Embedding with MLP Parameter Estimation

## Task Description

Implement a hybrid quantum-classical model with:
- **Linear Layers**: 2-3 layer MLP to estimate PQC parameters
- **PQC**: 4-5 qubit parameterized quantum circuit
- **Data**: Normally distributed input samples
- **Training**: MSE Loss

## Implementation

### Architecture
- **MLP**: 1 → 32 → 64 → 15 (ReLU activations)
- **PQC**: 5 qubits, 3 layers with RY rotations + ring CNOT entanglement
- **Output**: Learnable scaling layer for expectation values

### Target Function
For input `x ~ N(0,1)`: `y = [x, sin(x), cos(x), x²]`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python main.py --epochs 50 --lr 0.005

# Custom configuration
python main.py --num_samples 2048 --batch_size 32 --epochs 50
```

## Results

| Metric | Baseline | Our Model | Improvement |
|--------|----------|-----------|-------------|
| Test MSE | 0.5397 | **0.0541** | **90.0%** |

### Per-Target MSE
| Target | MSE |
|--------|-----|
| x | 0.0086 |
| sin(x) | 0.0023 |
| cos(x) | 0.0026 |
| x² | 0.2229 |

## Files

- `main.py` - Entry point for training
- `src/model.py` - Hybrid model and PQC definition
- `src/dataset.py` - Data generation utilities
- `src/training.py` - Training and evaluation functions
- `PLANNING.md` - Design decisions and research
- `DOCUMENTATION.md` - Detailed documentation

## Dependencies

- PyTorch >= 2.0.0
- PennyLane >= 0.33.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
