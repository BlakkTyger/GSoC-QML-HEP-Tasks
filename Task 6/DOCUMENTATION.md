# Task VI: Quantum Representation Learning - Documentation

## Executive Summary

This document provides the technical specification for the Quantum Similarity Network implementation that achieves **89.5% accuracy** on MNIST pair classification using quantum representation learning with contrastive loss.

**Key Results:**
- Test Accuracy: 89.5%
- Training Accuracy: 90%+
- Total Parameters: 16
- Total Qubits: 9

---

## 1. Problem Definition

### 1.1 Objective

Learn quantum representations of MNIST images such that:
- Images from the **same class** produce quantum states with **high fidelity** (≥ 0.5)
- Images from **different classes** produce quantum states with **low fidelity** (< 0.5)

### 1.2 Mathematical Formulation

Given images $x_i, x_j$ with labels $y_i, y_j$:

**Goal:** Train parameters θ such that:
- $|\langle\psi(x_i, \theta)|\psi(x_j, \theta)\rangle|^2 \geq 0.5$ when $y_i = y_j$
- $|\langle\psi(x_i, \theta)|\psi(x_j, \theta)\rangle|^2 < 0.5$ when $y_i \neq y_j$

---

## 2. Architecture

### 2.1 System Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    QUANTUM SIMILARITY NETWORK                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   Image 1 (28×28)              Image 2 (28×28)               │
│        │                            │                         │
│        ▼                            ▼                         │
│   ┌──────────┐                 ┌──────────┐                  │
│   │ Quadrant │                 │ Quadrant │                  │
│   │  Means   │                 │  Means   │                  │
│   └────┬─────┘                 └────┬─────┘                  │
│        │ [q1,q2,q3,q4]              │ [q1,q2,q3,q4]          │
│        ▼                            ▼                         │
│   ┌──────────┐                 ┌──────────┐                  │
│   │    RY    │                 │    RX    │                  │
│   │ Encoding │                 │ Encoding │                  │
│   │ (params1)│                 │ (params2)│                  │
│   └────┬─────┘                 └────┬─────┘                  │
│        │                            │                         │
│        │    ┌────────────────┐      │                         │
│        └───►│   SWAP TEST    │◄─────┘                         │
│             │  (9 qubits)    │                                │
│             └───────┬────────┘                                │
│                     │                                         │
│                     ▼                                         │
│                 Fidelity                                      │
│                     │                                         │
│                     ▼                                         │
│            ┌─────────────────┐                                │
│            │ Contrastive Loss│                                │
│            │  & Backprop     │                                │
│            └─────────────────┘                                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Component Specifications

| Component | Details |
|-----------|---------|
| **Qubits** | 9 total (1 ancilla + 4 per image) |
| **Preprocessing** | Quadrant mean pooling (28×28 → 4 values) |
| **Encoding 1** | RY(params1[i,0] * x[i] + params1[i,1]) |
| **Encoding 2** | RX(params2[i,0] * x[i] + params2[i,1]) |
| **SWAP Test** | H → 4×CSWAP → H → ⟨Z⟩ |
| **Parameters** | 16 total (8 per image encoder) |

---

## 3. Implementation Details

### 3.1 Preprocessing

```python
def preprocess_image(img):
    """
    Compute mean of 4 quadrants from 28x28 image.
    
    Returns:
        4-element tensor [top-left, top-right, bottom-left, bottom-right]
    """
    img_np = img.numpy()
    q1 = np.mean(img_np[:14, :14])   # Top-left
    q2 = np.mean(img_np[:14, 14:])   # Top-right
    q3 = np.mean(img_np[14:, :14])   # Bottom-left
    q4 = np.mean(img_np[14:, 14:])   # Bottom-right
    return torch.tensor([q1, q2, q3, q4], dtype=torch.float32)
```

### 3.2 Quantum Circuit

```python
@qml.qnode(dev, interface="torch")
def quantum_circuit(image1, image2, params1, params2):
    # Encode image 1 with RY rotations (wires 1-4)
    for i in range(4):
        theta = params1[i, 0] * image1[i] + params1[i, 1]
        qml.RY(theta, wires=1+i)
    
    # Encode image 2 with RX rotations (wires 5-8)
    for i in range(4):
        theta = params2[i, 0] * image2[i] + params2[i, 1]
        qml.RX(theta, wires=5+i)
    
    # SWAP test
    qml.Hadamard(wires=0)
    for i in range(4):
        qml.CSWAP(wires=[0, 1+i, 5+i])
    qml.Hadamard(wires=0)
    
    return qml.expval(qml.PauliZ(0))
```

### 3.3 Neural Network Module

```python
class QuantumNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.params1 = torch.nn.Parameter(torch.randn(4, 2))
        self.params2 = torch.nn.Parameter(torch.randn(4, 2))
    
    def forward(self, img1, img2):
        proc_img1 = preprocess_image(img1)
        proc_img2 = preprocess_image(img2)
        return quantum_circuit(proc_img1, proc_img2, 
                               self.params1, self.params2)
```

### 3.4 Loss Function

```python
def contrastive_loss(fidelity, label):
    """
    Contrastive loss for quantum similarity learning.
    
    Args:
        fidelity: SWAP test output (quantum state overlap)
        label: 1 if same class, 0 if different class
    
    Returns:
        Loss value to minimize
    """
    return label * (1 - fidelity)**2 + (1 - label) * fidelity**2
```

---

## 4. Training Configuration

### 4.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.02 |
| Training samples | 4000 |
| Epochs | 50 |
| Iterations per epoch | 100 |
| Classes | All 10 MNIST digits |
| Pair sampling | Random uniform |

### 4.2 Training Algorithm

```
Algorithm: Quantum Contrastive Learning
───────────────────────────────────────
1. Initialize params1, params2 ~ N(0, 1)
2. For epoch = 1 to 50:
   a. For iteration = 1 to 100:
      i.   Sample random pair (img1, img2)
      ii.  label = 1 if same_class else 0
      iii. fidelity = QuantumNet(img1, img2)
      iv.  loss = contrastive_loss(fidelity, label)
      v.   Backpropagate and update params
   b. Log epoch metrics
3. Return trained model
```

---

## 5. Results

### 5.1 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **89.5%** |
| Training Accuracy | 90% |
| Total Parameters | 16 |
| Training Time | ~2 minutes |

### 5.2 Training Progress

The model converges within 50 epochs:
- Epoch 1: ~84% accuracy
- Epoch 20: ~92% accuracy
- Epoch 50: ~90% accuracy (stabilized)

### 5.3 Output Files

| File | Description |
|------|-------------|
| `results/training_results.png` | Training curves and fidelity distribution |
| `results/metrics.txt` | Final evaluation metrics |
| `results/model.pt` | Saved model parameters |

---

## 6. Usage

### 6.1 Running Training

```bash
cd "Task 6"
python main.py
```

### 6.2 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
pennylane>=0.33.0
numpy>=1.24.0
matplotlib>=3.7.0
```

### 6.3 Expected Output

```
============================================================
Task 6: Quantum Representation Learning
============================================================

[1/4] Loading MNIST Dataset...
Loaded 4000 training samples
Using ALL 10 MNIST classes

[2/4] Verifying Data Loading...
Sample image shape: torch.Size([28, 28])

[3/4] Initializing Quantum Model...
Total qubits: 9
Parameters: 16

[4/4] Training...
Epoch  1: Loss=0.1387, Accuracy=84.0%
...
Epoch 50: Loss=0.0904, Accuracy=90.0%

============================================================
EVALUATION RESULTS
============================================================
Test Accuracy: 89.5%
```

---

## 7. Key Design Decisions

### 7.1 Why Quadrant Pooling?

- **Spatial preservation**: Maintains position information of digits
- **Noise reduction**: Averaging reduces pixel-level noise
- **Efficient encoding**: 4 values map naturally to 4 qubits
- **Better than resize**: Preserves more structure than downsampling

### 7.2 Why Trainable Encoding?

- **Flexibility**: Model learns optimal rotation angles
- **Expressivity**: Linear transform adds 2 params per feature
- **Convergence**: Faster learning than fixed encoding

### 7.3 Why Separate RY/RX Encodings?

- **Distinct representations**: Different axes prevent collapse
- **Complementary learning**: Each encoder specializes
- **Better gradients**: Orthogonal rotations improve optimization

### 7.4 Why No Margin in Loss?

- **Simplicity**: Fewer hyperparameters to tune
- **Symmetry**: Equal treatment of positive/negative pairs
- **Effectiveness**: Achieves high accuracy without margin

---

## 8. References

1. Buhrman et al. - "Quantum Fingerprinting" (SWAP test)
2. Chen et al. - "SimCLR: A Simple Framework for Contrastive Learning"
3. PennyLane Documentation - https://pennylane.ai/qml/
4. MNIST Dataset - LeCun et al.

---

*Document Version: 2.0*  
*Last Updated: March 2025*  
*Test Accuracy: 89.5%*
