# Task VIII: Vision Transformer / Quantum Vision Transformer

## Task Description
Implement a classical Vision Transformer and apply it to MNIST. Show its performance on the test data. Comment on potential ideas to extend this classical vision transformer architecture to a quantum vision transformer and sketch out the architecture in detail.

## Results Summary

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **97.58%** |
| Baseline (Task 8.ipynb) | 97.44% |
| Improvement | +0.14% |
| Parameters | 205,962 |
| Training Time | ~17 min (CPU) |

## Architecture

**Vision Transformer (ViT)** adapted for MNIST:
- **Patch Size**: 7×7 (16 patches from 28×28 image)
- **Embedding Dimension**: 64
- **Transformer Depth**: 6 blocks
- **Attention Heads**: 4
- **MLP Ratio**: 2.0

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py --epochs 15 --save-model

# Results are saved in results/
```

## Project Structure

```
Task 8/
├── src/                    # Source code
│   ├── model.py           # ViT architecture
│   ├── dataset.py         # MNIST data loading
│   ├── training.py        # Training pipeline
│   └── utils.py           # Utilities
├── results/               # Training outputs
│   ├── metrics.txt        # Performance metrics
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── vit_mnist.pt       # Saved model
├── main.py                # Entry point
├── PLANNING.md            # Planning document
├── DOCUMENTATION.md       # Full documentation
└── README.md              # This file
```

## Quantum Vision Transformer Proposal

The documentation includes a detailed proposal for extending to a Quantum ViT:

1. **Quantum MLP Layers** - Replace classical MLP with PQC (recommended)
2. **Quantum Attention** - Use SWAP test for similarity computation
3. **Quantum Classification Head** - VQC for final classification

See `DOCUMENTATION.md` for full architecture details and implementation sketches.

## Training Curves

![Training Curves](results/training_curves.png)

## Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

## References

- Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020)
- Cherrat et al., "Quantum Vision Transformers" (2022)
