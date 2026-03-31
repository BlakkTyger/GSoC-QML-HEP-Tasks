# Task VIII: Vision Transformer / Quantum Vision Transformer - Documentation

## 1. Executive Summary

This project implements a **Vision Transformer (ViT)** for MNIST digit classification and provides a detailed proposal for extending it to a **Quantum Vision Transformer (QViT)**.

### Key Results
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **97.58%** |
| Baseline Accuracy | 97.44% |
| Improvement | +0.14% |
| Model Parameters | 205,962 |
| Training Epochs | 15 |

---

## 2. Vision Transformer Architecture

### 2.1 Architecture Overview

The Vision Transformer adapts the Transformer architecture (originally designed for NLP) for image classification by treating images as sequences of patches.

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (28×28×1)                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    PATCH EMBEDDING                           │
│         Conv2d(1, 64, kernel=7, stride=7)                   │
│         Output: 16 patches of dimension 64                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              [CLS] TOKEN + POSITIONAL EMBEDDING              │
│         Sequence: 17 tokens × 64 dimensions                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                TRANSFORMER ENCODER (×6)                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Layer Norm → Multi-Head Attention (4 heads)         │    │
│  │ + Residual Connection                               │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Layer Norm → MLP (64 → 128 → 64)                    │    │
│  │ + Residual Connection                               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                CLASSIFICATION HEAD                           │
│         Layer Norm → Linear(64, 10)                         │
│         Uses [CLS] token output                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT (10 classes)                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

#### Patch Embedding
- **Input**: 28×28 grayscale image
- **Patch Size**: 7×7 pixels
- **Number of Patches**: 16 (4×4 grid)
- **Method**: Conv2d with kernel_size=stride=patch_size

#### Positional Encoding
- **Type**: Learnable 1D positional embeddings
- **Size**: 17 × 64 (16 patches + 1 CLS token)
- **Initialization**: Truncated normal (std=0.02)

#### Multi-Head Self-Attention
- **Heads**: 4
- **Head Dimension**: 16 (64/4)
- **Implementation**: PyTorch's `nn.MultiheadAttention` with `batch_first=True`

#### MLP Block
- **Architecture**: Linear → GELU → Dropout → Linear → Dropout
- **Expansion Ratio**: 2× (64 → 128 → 64)

### 2.3 Model Configuration

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
}
```

---

## 3. Training Details

### 3.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 0.01 |
| LR Scheduler | Cosine Annealing |
| Batch Size | 128 |
| Epochs | 15 |

### 3.2 Data Preprocessing

- **Normalization**: mean=0.1307, std=0.3081
- **Augmentation**: Random rotation (±10°)
- **Train/Val Split**: 90%/10%

### 3.3 Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 1.1664 | 64.13% | 0.5719 | 82.97% |
| 5 | 0.2143 | 93.44% | 0.1439 | 95.50% |
| 10 | 0.1306 | 95.92% | 0.1002 | 96.88% |
| 15 | 0.1071 | 96.64% | 0.0872 | 97.32% |

**Best Validation Accuracy**: 97.42% (Epoch 13)

---

## 4. Evaluation Results

### 4.1 Test Performance

- **Test Accuracy**: 97.58%
- **Test Loss**: 0.0732

### 4.2 Per-Class Performance

| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9749 | 0.9929 | 0.9838 | 980 |
| 1 | 0.9877 | 0.9903 | 0.9890 | 1135 |
| 2 | 0.9718 | 0.9671 | 0.9694 | 1032 |
| 3 | 0.9675 | 0.9713 | 0.9694 | 1010 |
| 4 | 0.9797 | 0.9807 | 0.9802 | 982 |
| 5 | 0.9675 | 0.9675 | 0.9675 | 892 |
| 6 | 0.9822 | 0.9802 | 0.9812 | 958 |
| 7 | 0.9673 | 0.9776 | 0.9724 | 1028 |
| 8 | 0.9802 | 0.9651 | 0.9726 | 974 |
| 9 | 0.9779 | 0.9633 | 0.9705 | 1009 |

**Macro Average F1**: 0.9756

### 4.3 Baseline Comparison

| Model | Test Accuracy | Parameters |
|-------|---------------|------------|
| Task 8.ipynb (baseline) | 97.44% | ~100K |
| QMLHEP_task_VIII.ipynb | 93.60% | ~2.9M |
| **Our ViT** | **97.58%** | 206K |

---

## 5. Quantum Vision Transformer Proposal

### 5.1 Motivation

Quantum computing offers potential advantages through:
1. **Quantum Superposition**: Process multiple states simultaneously
2. **Quantum Entanglement**: Capture complex correlations
3. **Exponential State Space**: 2^n states with n qubits

### 5.2 Proposed QViT Architecture

#### Option 1: Quantum MLP Layers (Recommended)

Replace classical MLP blocks with hybrid quantum-classical layers:

```
Classical Input (64 dim)
        ↓
Linear(64 → n_qubits)  [Dimensionality reduction]
        ↓
┌───────────────────────────────────────┐
│         QUANTUM CIRCUIT               │
│  ┌─────────────────────────────────┐  │
│  │  Angle Encoding                 │  │
│  │  RY(x₀), RY(x₁), ..., RY(xₙ₋₁)  │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │  Variational Layer 1            │  │
│  │  RY(θ), RZ(θ) + CNOT ladder     │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │  Variational Layer 2            │  │
│  │  RY(θ), RZ(θ) + CNOT ladder     │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │  Measurement: ⟨Z⟩ expectations  │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘
        ↓
Linear(n_qubits → 64)  [Dimensionality expansion]
        ↓
Classical Output (64 dim)
```

**Implementation Sketch (PennyLane)**:
```python
import pennylane as qml

n_qubits = 8
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_mlp(inputs, weights):
    # Angle encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Variational layers
    for layer in range(n_layers):
        # Rotation gates
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        # Entanglement (CNOT ladder)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    # Measurements
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumMLP(nn.Module):
    def __init__(self, input_dim, n_qubits, n_layers):
        super().__init__()
        self.pre_net = nn.Linear(input_dim, n_qubits)
        self.post_net = nn.Linear(n_qubits, input_dim)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
    
    def forward(self, x):
        x = torch.tanh(self.pre_net(x)) * np.pi  # Scale to [-π, π]
        x = quantum_mlp(x, self.weights)
        x = torch.stack(x, dim=-1)
        return self.post_net(x)
```

#### Option 2: Quantum Attention

Replace classical attention with quantum kernel-based similarity:

```
Query, Key encoded as quantum states
        ↓
SWAP test for similarity measurement
        ↓
Softmax over similarity scores
        ↓
Classical weighted sum of Values
```

**Challenges**:
- Requires O(N²) SWAP tests for N patches
- Limited by qubit count for embedding dimension
- Slower than classical attention for current hardware

#### Option 3: Quantum Classification Head

Simplest approach - replace only the final classifier:

```
[CLS] token (64 dim)
        ↓
Linear(64 → n_qubits)
        ↓
VQC with data re-uploading
        ↓
10 expectation values → 10 classes
```

### 5.3 Recommended QViT Configuration

```python
qvit_config = {
    # Classical components (unchanged)
    'img_size': 28,
    'patch_size': 7,
    'embed_dim': 64,
    'num_heads': 4,
    
    # Quantum components
    'quantum_mlp': True,
    'n_qubits': 8,
    'n_quantum_layers': 2,
    'quantum_head': True,
    
    # Hybrid settings
    'classical_depth': 4,  # Reduced from 6
    'quantum_depth': 2,    # Quantum MLP in last 2 blocks
}
```

### 5.4 Expected Trade-offs

| Aspect | Classical ViT | Quantum ViT |
|--------|---------------|-------------|
| Training Speed | Fast | Slow (simulation) |
| Parameters | ~206K | ~50K + quantum |
| Expressivity | Limited by width | Exponential in qubits |
| Hardware | GPU/CPU | NISQ devices |
| Scalability | Linear | Limited by qubits |

### 5.5 Research Directions

1. **Quantum Advantage**: Investigate tasks where QViT outperforms classical ViT
2. **Noise Resilience**: Develop error mitigation strategies for NISQ devices
3. **Efficient Encoding**: Explore amplitude encoding for higher-dimensional inputs
4. **Hybrid Training**: Combine quantum and classical gradient computation

---

## 6. Project Structure

```
Task 8/
├── src/
│   ├── __init__.py          # Package exports
│   ├── model.py             # ViT architecture
│   ├── dataset.py           # MNIST data loading
│   ├── training.py          # Training pipeline
│   └── utils.py             # Utilities and visualization
├── results/
│   ├── metrics.txt          # Performance metrics
│   ├── training_curves.png  # Loss/accuracy plots
│   ├── confusion_matrix.png # Confusion matrix
│   ├── classification_report.txt
│   └── vit_mnist.pt         # Saved model
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── PLANNING.md              # Planning document
├── DOCUMENTATION.md         # This file
└── README.md                # Quick start guide
```

---

## 7. Usage

### Training
```bash
# Full training with validation
python main.py --epochs 15 --save-model

# Simple training (no validation split)
python main.py --simple --epochs 10
```

### Inference
```python
import torch
from src.model import create_vit

# Load model
model = create_vit(config)
checkpoint = torch.load('results/vit_mnist.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1)
```

---

## 8. References

1. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv:2010.11929
2. Cherrat, E. A., et al. (2022). "Quantum Vision Transformers." arXiv:2209.08167
3. Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS 2017
4. Schuld, M., & Petruccione, F. (2021). "Machine Learning with Quantum Computers." Springer

---

*Document Version: 1.0*
*Last Updated: Implementation Complete*
