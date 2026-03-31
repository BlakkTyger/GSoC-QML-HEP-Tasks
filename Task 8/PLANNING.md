# Task VIII: Vision Transformer / Quantum Vision Transformer - Planning Document

## 1. Task Overview

**Objective:** Implement a classical Vision Transformer (ViT) for MNIST classification, achieve competitive accuracy (≥97.44% baseline), and propose a detailed Quantum Vision Transformer (QViT) architecture.

**Deliverables:**
1. Modular ViT implementation with training/evaluation pipeline
2. Test accuracy matching or exceeding baseline (97.44%)
3. Detailed QViT architecture proposal with implementation sketch

---

## 2. Baseline Analysis

### 2.1 Task 8.ipynb (Simple ViT)
| Parameter | Value |
|-----------|-------|
| Image Size | 28×28 |
| Patch Size | 7 |
| Number of Patches | 16 (4×4) |
| Embed Dimension | 64 |
| Depth (Transformer Blocks) | 6 |
| Attention Heads | 4 |
| MLP Ratio | 2.0 |
| Dropout | 0.1 |
| Learning Rate | 3e-4 |
| Epochs | 10 |
| **Test Accuracy** | **97.44%** |

**Key Observations:**
- Uses `nn.MultiheadAttention` (efficient implementation)
- Simple patch embedding via Conv2d
- Class token + positional embeddings
- Lightweight architecture (~100K parameters)

### 2.2 QMLHEP_task_VIII.ipynb (Complex ViT)
| Parameter | Value |
|-----------|-------|
| Patch Size | 4 |
| Number of Patches | 49 (7×7) |
| Embed Dimension | 256 |
| Layers | 6 |
| Attention Heads | 8 |
| Forward Multiplier | 2 |
| **Test Accuracy** | **93.6%** |
| Parameters | ~2.9M |

**Key Observations:**
- Custom self-attention implementation
- Uses PyTorch Lightning
- Larger model but lower accuracy (possibly due to early stopping/keyboard interrupt)
- More complex classifier head (Linear → Tanh → Linear)

### 2.3 Target Baseline
**Primary target: ≥97.44% test accuracy** (from Task 8.ipynb)

---

## 3. Dataset Understanding: MNIST

### 3.1 Dataset Characteristics
- **Images:** 28×28 grayscale images of handwritten digits (0-9)
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images
- **Classes:** 10 (digits 0-9)
- **Pixel Values:** 0-255 (normalized to [0,1] or standardized)

### 3.2 MNIST Statistics
- Mean: 0.1307
- Std: 0.3081

### 3.3 Why ViT for MNIST?
Traditional CNNs excel at MNIST, but ViT demonstrates:
1. **Global context learning** - Attention can capture long-range dependencies
2. **Patch-based representation** - Each patch represents local features
3. **Scalability** - Architecture scales to larger images/datasets
4. **Foundation for quantum extension** - Attention mechanism can be quantized

---

## 4. Vision Transformer Architecture Research

### 4.1 Original ViT Paper (Dosovitskiy et al., 2020)
**"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**

**Core Components:**
1. **Patch Embedding:** Split image into fixed-size patches, flatten, linearly project
2. **Position Embedding:** Learnable 1D positional embeddings
3. **Class Token:** Prepended learnable [CLS] token for classification
4. **Transformer Encoder:** Standard encoder with Multi-Head Self-Attention + MLP
5. **Classification Head:** MLP on [CLS] token output

### 4.2 Architecture Formula
For image of size H×W with patch size P:
- Number of patches: N = (H/P) × (W/P)
- Patch dimension: P² × C (where C is channels)
- Sequence length: N + 1 (including [CLS] token)

For MNIST (28×28, 1 channel):
- Patch size 7 → 16 patches, patch dim = 49
- Patch size 4 → 49 patches, patch dim = 16
- Patch size 14 → 4 patches, patch dim = 196

### 4.3 Key Design Choices for MNIST

| Choice | Rationale |
|--------|-----------|
| Patch Size = 7 | Divides 28 evenly, 16 patches is manageable |
| Embed Dim = 64-128 | Small images don't need huge embeddings |
| Depth = 4-6 | Sufficient for simple digit patterns |
| Heads = 4-8 | Must divide embed_dim evenly |
| MLP Ratio = 2-4 | Standard expansion factor |

### 4.4 Attention Mechanism Deep Dive

**Multi-Head Self-Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- Q, K, V are linear projections of input
- d_k is the key dimension (embed_dim / num_heads)
- Output is concatenation of all heads passed through output projection

**Why Self-Attention for Images?**
- Captures global dependencies (pixel at corner can attend to center)
- Learns spatial relationships without explicit convolution
- Position-agnostic (position info via embeddings)

---

## 5. Quantum Vision Transformer Research

### 5.1 Literature Review

**Key Papers:**
1. **"Quantum Vision Transformers" (Cherrat et al., 2022)** - First QViT proposal
2. **"Quantum Convolutional Neural Networks" (Cong et al., 2019)** - Quantum convolution foundations
3. **"Variational Quantum Classifiers" (Schuld et al., 2020)** - VQC for classification

### 5.2 Quantum Computing Constraints

| Constraint | Impact on Design |
|------------|------------------|
| Limited qubits (NISQ era) | Must reduce dimensionality before quantum layers |
| Noisy gates | Use shallow circuits, error mitigation |
| Slow simulation | Hybrid approach preferable |
| Barren plateaus | Careful initialization, local cost functions |

### 5.3 Potential Quantum Extensions

#### Option A: Quantum MLP Layers (Recommended)
**Location:** Replace MLP blocks in Transformer encoder with PQC

**Architecture:**
```
Input (embed_dim) → Linear (embed_dim → n_qubits) → PQC → Linear (n_qubits → embed_dim)
```

**Advantages:**
- Minimal architectural change
- Can use existing attention mechanism
- Scalable to different qubit counts

#### Option B: Quantum Attention
**Location:** Replace attention computation with quantum kernel/SWAP test

**Architecture:**
```
Q, K encoded as quantum states → SWAP test for similarity → Classical softmax → V
```

**Challenges:**
- Requires encoding all patches as quantum states
- Multiple measurements needed (one per attention weight)
- Slow for many patches

#### Option C: Quantum Patch Embedding
**Location:** Replace Conv2d patch embedding with quantum convolution

**Architecture:**
```
Image patches → Quantum kernel convolution → Classical features
```

**Based on:** Quantum Convolutional Neural Networks (Cong et al.)

#### Option D: Quantum Classification Head (Simplest)
**Location:** Replace final MLP head with VQC

**Architecture:**
```
[CLS] token (embed_dim) → Linear (embed_dim → n_qubits) → VQC → 10 outputs
```

**Advantages:**
- Simplest implementation
- Minimal quantum resources
- Easy to benchmark

### 5.4 Recommended QViT Architecture

**Hybrid Approach: Quantum-Enhanced MLP + Quantum Head**

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (28×28)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              CLASSICAL PATCH EMBEDDING                       │
│         Conv2d(1, embed_dim, patch_size, patch_size)        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              [CLS] TOKEN + POSITIONAL EMBEDDING              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  QUANTUM TRANSFORMER BLOCK                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ LayerNorm → Multi-Head Self-Attention → Residual    │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ LayerNorm → QUANTUM MLP (PQC) → Residual            │    │
│  │                                                      │    │
│  │  Linear(embed→n_qubits) → VQC → Linear(n_qubits→embed)│   │
│  └─────────────────────────────────────────────────────┘    │
│                     × depth layers                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 QUANTUM CLASSIFICATION HEAD                  │
│          [CLS] → Linear(embed→n_qubits) → VQC → 10          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       OUTPUT (10 classes)                    │
└─────────────────────────────────────────────────────────────┘
```

### 5.5 PQC Design for QViT

**Quantum Circuit Structure:**
```
     ┌───────────────────────────────────────────────────┐
     │  DATA ENCODING (Angle Embedding)                  │
     │  RY(x_0), RY(x_1), ..., RY(x_{n-1})              │
     ├───────────────────────────────────────────────────┤
     │  VARIATIONAL LAYER 1                              │
     │  RY(θ_0), RZ(θ_1), ... + CNOT entanglement       │
     ├───────────────────────────────────────────────────┤
     │  VARIATIONAL LAYER 2                              │
     │  RY(θ_k), RZ(θ_{k+1}), ... + CNOT entanglement   │
     ├───────────────────────────────────────────────────┤
     │  ... (L layers total)                             │
     ├───────────────────────────────────────────────────┤
     │  MEASUREMENT                                      │
     │  <Z_0>, <Z_1>, ..., <Z_{n-1}>                    │
     └───────────────────────────────────────────────────┘
```

**Entanglement Patterns:**
- Linear: CNOT(i, i+1) for all i
- Circular: Linear + CNOT(n-1, 0)
- All-to-all: CNOT for all pairs (expensive)

---

## 6. Implementation Plan

### 6.1 Classical ViT Implementation

**File Structure:**
```
Task 8/
├── src/
│   ├── __init__.py
│   ├── model.py          # ViT model definition
│   ├── dataset.py        # MNIST data loading
│   ├── training.py       # Training loop
│   └── utils.py          # Utilities
├── results/
│   ├── metrics.txt       # Final metrics
│   ├── training_curve.png
│   └── confusion_matrix.png
├── main.py               # Entry point
├── requirements.txt
├── PLANNING.md
├── DOCUMENTATION.md
└── README.md
```

### 6.2 Model Configuration

**Target Configuration (based on baseline analysis):**
```python
config = {
    'img_size': 28,
    'patch_size': 7,
    'in_channels': 1,
    'num_classes': 10,
    'embed_dim': 64,
    'depth': 6,
    'num_heads': 4,
    'mlp_ratio': 2.0,
    'dropout': 0.1,
    'learning_rate': 3e-4,
    'batch_size': 128,
    'epochs': 15
}
```

### 6.3 Training Strategy

1. **Optimizer:** AdamW with weight decay
2. **Learning Rate:** 3e-4 with cosine annealing
3. **Data Augmentation:** Minimal (random rotation ±10°) - MNIST is simple
4. **Regularization:** Dropout (0.1), weight decay
5. **Early Stopping:** Patience of 5 epochs on validation loss

### 6.4 Evaluation Metrics

- Test Accuracy (primary metric)
- Training/Validation Loss curves
- Confusion Matrix
- Per-class Precision/Recall

---

## 7. Risk Assessment & Mitigations

| Risk | Mitigation |
|------|------------|
| Overfitting | Dropout, early stopping, data augmentation |
| Underfitting | Increase model capacity, more epochs |
| Slow training | Use GPU, optimize batch size |
| Low accuracy | Hyperparameter tuning, architecture search |

---

## 8. Timeline

1. **Phase 1:** Implement core ViT model (~1 hour)
2. **Phase 2:** Training pipeline and evaluation (~30 min)
3. **Phase 3:** Train and optimize (~1 hour)
4. **Phase 4:** Documentation and QViT proposal (~30 min)

---

## 9. References

1. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv:2010.11929
2. Cherrat, E. A., et al. (2022). "Quantum Vision Transformers." arXiv:2209.08167
3. Cong, I., Choi, S., & Lukin, M. D. (2019). "Quantum convolutional neural networks." Nature Physics, 15(12), 1273-1278.
4. Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS 2017.

---

## 10. Architecture Decision Log

### Decision 1: Patch Size = 7
**Rationale:** Divides 28 evenly into 16 patches. Larger patches (14) give too few patches for effective attention. Smaller patches (4) create 49 patches which increases computation without significant accuracy gain on MNIST.

### Decision 2: Embed Dimension = 64
**Rationale:** MNIST is simple; larger embeddings don't improve accuracy but increase parameters. The baseline achieved 97.44% with embed_dim=64.

### Decision 3: Use nn.MultiheadAttention
**Rationale:** PyTorch's optimized implementation is faster than custom attention. No need for custom implementation unless adding quantum components.

### Decision 4: Hybrid QViT over Full QViT
**Rationale:** Full quantum ViT is impractical with NISQ devices. Hybrid approach maintains classical attention (which works well) while introducing quantum components where they can provide advantage.

---

*Document Version: 1.0*
*Last Updated: During Implementation*
