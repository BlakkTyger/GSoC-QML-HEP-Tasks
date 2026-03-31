# Task VI: Quantum Representation Learning

## Overview

Quantum Similarity Network using SWAP test for contrastive learning on MNIST digits.

## Final Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 89.5% |
| **Training Accuracy** | 90% |
| **Parameters** | 16 |
| **Qubits** | 9 (1 ancilla + 4 + 4) |

## Architecture

```
Image (28×28) → Quadrant Means (4 values) → Trainable Encoding → SWAP Test → Fidelity
```

- **Preprocessing**: Quadrant mean pooling (28×28 → 4 features)
- **Encoding**: θ = params[i,0] * x[i] + params[i,1]
  - Image 1: RY rotations
  - Image 2: RX rotations
- **Measurement**: SWAP test for fidelity
- **Loss**: Contrastive loss: `label*(1-fid)² + (1-label)*fid²`

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Project Structure

```
Task 6/
├── main.py              # Complete implementation
├── requirements.txt     # Dependencies
├── PLANNING.md          # Research & design document
├── DOCUMENTATION.md     # Technical documentation
├── data/MNIST/          # Downloaded dataset
└── results/
    ├── metrics.txt      # Final metrics
    ├── model.pt         # Trained parameters
    └── training_results.png
```

## Task Requirements

- Load the MNIST dataset ✓
- Trainable quantum state preparation ✓
- Dual-image SWAP test circuit ✓
- Contrastive loss training ✓