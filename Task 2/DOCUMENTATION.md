# Task II: Classical GNN for Quark/Gluon Classification - Documentation

## Executive Summary

This project implements two Graph Neural Network architectures for classifying particle jets as originating from quarks or gluons. The task uses the ParticleNet quark/gluon dataset, accessible via the EnergyFlow Python package.

**Selected Architectures:**
1. **ParticleNet (Dynamic EdgeConv)** - State-of-the-art architecture with dynamic graph learning
2. **GAT Classifier (GATv2Conv)** - Attention-based architecture with interpretable weights

---

## 1. Problem Overview

### 1.1 Physics Context

Quark/gluon jet classification is a fundamental problem in high-energy physics:
- **Jets** are collimated sprays of hadrons produced when quarks/gluons hadronize
- **Gluon jets** have higher particle multiplicity and are broader due to larger color charge (CA = 3)
- **Quark jets** are more collimated with harder fragmentation (CF = 4/3)
- Accurate classification improves background rejection in Higgs and BSM searches

### 1.2 Dataset

| Property | Value |
|----------|-------|
| Source | Zenodo (via EnergyFlow) |
| Jets per class | 50,000 quark + 50,000 gluon |
| Features per particle | (pT, rapidity η, azimuthal angle φ, pdgid) |
| Jet pT range | 500-550 GeV |
| Jet clustering | anti-kT, R=0.4 |

---

## 2. Graph Construction

### 2.1 Point Cloud to Graph Transformation

Jets are naturally **point clouds** - unordered sets of particles. We transform them to graphs:

**Nodes:** Each particle becomes a node
**Edges:** k-Nearest Neighbors (k=16) in (Δη, Δφ) space

### 2.2 Feature Engineering

**Node Features (5 dimensions):**
| Feature | Description | Motivation |
|---------|-------------|------------|
| Δη | Rapidity relative to jet centroid | Spatial position |
| Δφ | Azimuthal angle relative to centroid | Spatial position |
| log(pT) | Log transverse momentum | Compresses dynamic range |
| log(E) | Log energy | Energy scale |
| pT/ΣpT | Momentum fraction | Relative importance |

**Edge Features (3 dimensions, for GAT):**
| Feature | Description |
|---------|-------------|
| ΔR_ij | Angular distance between particles |
| Δη_ij | Rapidity difference |
| Δφ_ij | Azimuthal difference |

### 2.3 Why k-NN Graphs?

1. **Locality:** Nearby particles in (η,φ) space are physically correlated
2. **Efficiency:** Fixed number of edges enables efficient batching
3. **Proven:** Used successfully in ParticleNet paper
4. **k=16:** Empirically optimal value from literature

---

## 3. Architecture Details

### 3.1 ParticleNet (Dynamic EdgeConv)

**Core Innovation:** Dynamic graph construction - k-NN is recomputed in **learned feature space** after each layer, allowing the model to learn optimal graph structure.

```
Architecture:
┌─────────────────────────────────────────────────┐
│ Input: (N, 5) node features                     │
├─────────────────────────────────────────────────┤
│ BatchNorm1d(5)                                  │
├─────────────────────────────────────────────────┤
│ EdgeConv Block 1: 5 → 64 (k-NN in feature space)│
│   └─ MLP(10→64→64→64) + BatchNorm + ReLU        │
├─────────────────────────────────────────────────┤
│ EdgeConv Block 2: 64 → 128                      │
│   └─ MLP(128→128→128→128) + BatchNorm + ReLU    │
├─────────────────────────────────────────────────┤
│ EdgeConv Block 3: 128 → 256                     │
│   └─ MLP(256→256→256→256) + BatchNorm + ReLU    │
├─────────────────────────────────────────────────┤
│ Global Pooling: mean ⊕ max → 512                │
├─────────────────────────────────────────────────┤
│ Classifier: 512 → 256 → 128 → 2                 │
│   └─ Linear + BN + ReLU + Dropout(0.3)          │
└─────────────────────────────────────────────────┘
```

**Key Operations:**
```python
# EdgeConv message function
h_i' = max_{j∈N(i)} MLP(h_i || (h_j - h_i))
```
- Concatenates source features with edge features (difference)
- Max aggregation captures most salient neighbor information

**Parameters:** ~500K trainable

### 3.2 GAT Classifier (GATv2Conv)

**Core Innovation:** Attention mechanism learns which neighbors are most important, providing interpretability.

```
Architecture:
┌─────────────────────────────────────────────────┐
│ Input: (N, 5) node features + (E, 3) edge attr  │
├─────────────────────────────────────────────────┤
│ BatchNorm1d(5)                                  │
├─────────────────────────────────────────────────┤
│ GATv2Conv Block 1: 5 → 64×4 heads = 256         │
│   └─ Multi-head attention + BN + ReLU           │
├─────────────────────────────────────────────────┤
│ GATv2Conv Block 2: 256 → 64×4 heads = 256       │
│   └─ Multi-head attention + BN + ReLU           │
├─────────────────────────────────────────────────┤
│ GATv2Conv Block 3: 256 → 64 (concat=False)      │
│   └─ Multi-head attention averaged              │
├─────────────────────────────────────────────────┤
│ Global Pooling: mean ⊕ max → 128                │
├─────────────────────────────────────────────────┤
│ Classifier: 128 → 128 → 64 → 2                  │
│   └─ Linear + BN + ReLU + Dropout(0.3)          │
└─────────────────────────────────────────────────┘
```

**Key Operations:**
```python
# GATv2 attention computation
α_ij = softmax_j(a^T · LeakyReLU(W[h_i || h_j]))
h_i' = Σ_j α_ij · W · h_j
```
- **GATv2** fixes the static attention problem of original GAT
- Attention weights α_ij are interpretable
- Edge features incorporated via edge_dim parameter

**Parameters:** ~300K trainable

---

## 4. Comparison: ParticleNet vs GAT

| Aspect | ParticleNet | GAT |
|--------|-------------|-----|
| **Graph Type** | Dynamic (recomputed each layer) | Static (fixed k-NN) |
| **Message Passing** | MLP + max aggregation | Attention-weighted sum |
| **Expressiveness** | High (learns structure) | High (adaptive weights) |
| **Interpretability** | Low | High (attention visualization) |
| **Computational Cost** | Higher (k-NN overhead) | Lower |
| **Edge Features** | Implicit (differences) | Explicit (ΔR, Δη, Δφ) |

### Expected Performance

Based on literature and architecture design:
- **ParticleNet:** AUC ~0.83-0.85 (state-of-the-art)
- **GAT:** AUC ~0.81-0.83 (competitive baseline)

---

## 5. Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Train both models
python train.py --model both --epochs 50 --batch_size 128

# Train only ParticleNet
python train.py --model particlenet --epochs 50

# Train only GAT
python train.py --model gat --epochs 50
```

### Evaluation

```bash
# Evaluate ParticleNet
python evaluate.py --model particlenet --checkpoint results/particlenet_best.pt --visualize

# Evaluate GAT
python evaluate.py --model gat --checkpoint results/gat_best.pt --visualize
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | both | Model to train (particlenet/gat/both) |
| `--num_data` | 100000 | Number of jets to load |
| `--batch_size` | 128 | Batch size |
| `--epochs` | 50 | Maximum epochs |
| `--lr` | 1e-3 | Learning rate |
| `--k` | 16 | k-NN neighbors |
| `--dropout` | 0.3 | Dropout probability |
| `--patience` | 10 | Early stopping patience |

---

## 6. Code Structure

```
Task 2/
├── PLANNING.md              # Detailed planning and research
├── DOCUMENTATION.md         # This document
├── README.md                # Task description
├── requirements.txt         # Dependencies
│
├── src/
│   ├── data/
│   │   ├── dataset.py       # JetGraphDataset, data loading
│   │   └── preprocessing.py # Feature engineering, graph construction
│   │
│   ├── models/
│   │   ├── particle_net.py  # ParticleNet (EdgeConv) implementation
│   │   └── gat_classifier.py # GAT implementation
│   │
│   ├── training/
│   │   ├── trainer.py       # Training loop with early stopping
│   │   └── metrics.py       # AUC, accuracy, ROC curves
│   │
│   └── utils/
│       └── visualization.py # Jet plots, attention visualization
│
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
└── results/                 # Saved models and plots
```

---

## 7. Key Design Decisions

### 7.1 Why These Two Architectures?

| Decision | Rationale |
|----------|-----------|
| **ParticleNet** | State-of-the-art on this exact task, dynamic graph learning is unique |
| **GAT** | Provides interpretability, alternative paradigm (attention vs max-pool) |
| **Not GIN** | Lacks edge features, less interpretable |
| **Not Transformer** | O(n²) scaling problematic for high-multiplicity jets |
| **Not LorentzNet** | Complex, narrow pT window reduces need for equivariance |

### 7.2 Graph Construction Choices

| Choice | Alternative | Rationale |
|--------|-------------|-----------|
| k-NN (k=16) | Fully connected | Captures local structure efficiently |
| (Δη, Δφ) space | Learned space | Physics-motivated, interpretable |
| Static (GAT) / Dynamic (PN) | - | Different approaches for comparison |

### 7.3 Feature Choices

| Feature | Rationale |
|---------|-----------|
| log(pT), log(E) | Compresses large dynamic range |
| pT/ΣpT | Captures relative importance |
| Centered coordinates | Translation invariance |

---

## 8. References

1. **ParticleNet:** Qu & Gouskos, "ParticleNet: Jet Tagging via Particle Clouds", arXiv:1902.08570
2. **DGCNN:** Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", arXiv:1801.07829
3. **GAT:** Veličković et al., "Graph Attention Networks", arXiv:1710.10903
4. **GATv2:** Brody et al., "How Attentive are Graph Attention Networks?", arXiv:2105.14491
5. **EnergyFlow:** Komiske et al., "Energy Flow Networks", arXiv:1810.05165
6. **PyTorch Geometric:** Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric"

---

*Document version: 1.0*
*Created for GSoC 2026 QML-HEP Task II*
