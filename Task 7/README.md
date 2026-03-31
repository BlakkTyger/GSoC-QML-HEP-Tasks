# Task VII: Z₂ × Z₂ Equivariant Quantum Neural Networks

## Overview

This task implements **equivariant quantum neural networks** that respect the Z₂ × Z₂ (Klein four-group) symmetry. We compare a standard variational quantum classifier against a symmetry-aware equivariant QNN on a binary classification task.

## Final Results

| Model | Train Accuracy | Test Accuracy | Parameters |
|-------|----------------|---------------|------------|
| **Standard QNN** | 91.56% | 88.75% | 12 |
| **Equivariant QNN** | 91.56% | 88.75% | 9 |

**Key Finding**: Equivariant QNN achieves same accuracy with **25% fewer parameters**.

## The Z₂ × Z₂ Symmetry

The Klein four-group consists of four transformations on 2D data (x₁, x₂):

| Transformation | Action | Physical Meaning |
|----------------|--------|------------------|
| Identity | (x₁, x₂) → (x₁, x₂) | No change |
| Exchange | (x₁, x₂) → (x₂, x₁) | Mirror along y=x |

The dataset classification rule `|x₁ - x₂| >= threshold` is inherently symmetric under coordinate exchange.

## Project Structure

```
Task 7/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── dataset.py            # Z₂×Z₂ symmetric dataset generation
│   ├── models.py             # Standard and Equivariant QNN models
│   ├── training.py           # Training loops & optimization
│   └── visualization.py      # Plotting utilities
├── results/
│   ├── figures/              # Generated plots
│   └── metrics.txt           # Final metrics
├── main.py                   # Main execution script
├── requirements.txt          # Dependencies
├── PLANNING.md               # Research & planning document
└── DOCUMENTATION.md          # Technical documentation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full comparison
python main.py --save

# Run with custom parameters
python main.py --epochs 100 --n_points 300 --lr 0.01 --save
```

## Architecture

### Standard QNN
- AngleEmbedding + BasicEntanglerLayers
- Independent parameters per qubit
- 12 trainable parameters

### Equivariant QNN
- AngleEmbedding + Tied RX rotations + CNOT
- Same rotation angle applied to both qubits (enforces symmetry)
- 9 trainable parameters (25% reduction)

## References

1. Meyer et al., "Exploiting symmetry in variational quantum machine learning" [arXiv:2205.06217](https://arxiv.org/abs/2205.06217)
2. Nguyen et al., "Theory for Equivariant Quantum Neural Networks" [arXiv:2210.08566](https://arxiv.org/abs/2210.08566)
