# GSoC 2026 – QMLHEP Tasks and Proposal

This repository contains my implementation for the **Quantum Machine Learning for High Energy Physics (QMLHEP)** screening tasks for Google Summer of Code 2026.

## Problem Statement

You can view the official problem statement [here](https://docs.google.com/document/d/1imoMEyC0r5IESonwgA7BThEQWDfdrOsoyfMfyJgyXmU/edit?tab=t.0).

---

## Repository Overview

| Task | Title | Approach | Key Result | Stack |
|---|---|---|---|---|
| **I** | Quantum Computing Foundations | Notebook-based exercises covering quantum gates, circuits, and basic QML concepts | — | Qiskit |
| **II** | Quark/Gluon Jet Classification with GNNs | ParticleNet (Dynamic EdgeConv) + GAT (GATv2Conv) on jet point clouds | AUC ~0.83–0.85 | PyTorch Geometric, EnergyFlow |
| **III** | Open Task | Essay on QGNNs, QCGNN, sparse graph construction, and PennyLane | — | — |
| **IV** | Quantum GAN for HEP Event Classification | VQC discriminator with systematic hyperparameter tuning (22 configs) | AUC 0.793 | Cirq, TFQ |
| **V** | Quantum Graph Neural Network (QGNN) | 4-qubit VQC with physics-motivated jet features and all-to-all CZ entanglement | AUC 0.854, 78% acc | Cirq |
| **VI** | Quantum Representation Learning | SWAP-test based contrastive learning on MNIST with trainable encoding | 89.5% acc (16 params) | PennyLane, PyTorch |
| **VII** | Z₂×Z₂ Equivariant QNN | Symmetry-aware vs standard QNN on Z₂×Z₂ symmetric binary classification | 88.75% acc, 25% param reduction | PennyLane, PyTorch |
| **VIII** | Vision Transformer (ViT) + Quantum ViT Proposal | Classical ViT for MNIST with detailed Quantum ViT design proposal | 97.58% acc | PyTorch |
| **IX** | Kolmogorov-Arnold Network (KAN) + QKAN Proposal | Gaussian basis KAN for MNIST with Quantum KAN design proposal | 97.17% acc | PyTorch |
| **XI** | PQC Embedding with MLP | MLP maps scalar inputs to PQC rotation angles; learnable output scaling | MSE 0.054 (90% improvement) | PennyLane, PyTorch |
| **XII** | PQC Embedding with RL (PPO) | Reimplements Task XI with Proximal Policy Optimization | MSE 1.04 | PennyLane, PyTorch |

---

## Repository Structure

```
GSoC-QML-HEP-Tasks/
├── README.md                 # This file
│
├── Task 1/                   # Quantum computing foundations (Jupyter notebook)
├── Task 2/                   # GNN jet classification (ParticleNet + GAT)
├── Task 3/                   # Open task essay
├── Task 4/                   # Quantum GAN / VQC for HEP classification
├── Task 5/                   # QGNN for quark/gluon jet tagging
├── Task 6/                   # Quantum similarity network (SWAP test)
├── Task 7/                   # Equivariant QNN (Z₂×Z₂ symmetry)
├── Task 8/                   # Vision Transformer for MNIST
├── Task 9/                   # Kolmogorov-Arnold Network for MNIST
├── Task 11/                  # PQC embedding with supervised learning
├── Task 12/                  # PQC embedding with reinforcement learning

```

Each task folder contains its own `README.md` with detailed documentation including problem statement, architecture, usage instructions, results, and references.

---

## Setup

Most tasks are self-contained Python projects. To get started with any task:

```bash
cd "Task X"
pip install -r requirements.txt
```

### Common Dependencies

| Package | Used In | Purpose |
|---|---|---|
| PyTorch ≥ 2.0 | Tasks 2, 6–9, 11–12 | Deep learning framework |
| PennyLane ≥ 0.33 | Tasks 6, 7, 11, 12 | Quantum circuit simulation + autodiff |
| Cirq ≥ 1.0 | Tasks 4, 5 | Quantum circuit construction |
| TensorFlow Quantum | Task 4 | Hybrid quantum-classical training |
| PyTorch Geometric ≥ 2.3 | Task 2 | Graph neural network layers |
| EnergyFlow ≥ 1.3 | Tasks 2, 5 | Quark/gluon jet dataset |

> **Note**: Python 3.9+ is recommended. TensorFlow Quantum has specific version compatibility requirements — see the [TFQ install guide](https://www.tensorflow.org/quantum/install).