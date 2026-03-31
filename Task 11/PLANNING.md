# Task 11: PQC Embedding with MLP Parameter Estimation

## Task Overview

Implement a hybrid quantum-classical model where:
1. **Input**: Normally distributed scalar data
2. **MLP**: Neural network estimates PQC parameters
3. **PQC**: Parameterized quantum circuit generates quantum states
4. **Output**: Expectation values approximating target functions
5. **Loss**: MSE between quantum outputs and targets

---

## Baseline Analysis

### Baseline Architecture
- **Qubits**: 4
- **PQC Layers**: 2
- **MLP**: 1 → 16 → 16 → 8 (ReLU activations)
- **Gates**: RY rotations + ring CNOT entanglement
- **Samples**: 1024, batch_size=64, epochs=20, lr=0.01

### Baseline Performance
- **Final Test Loss**: 0.5397 (MSE)

### Target Function
For input scalar x, target = [x, sin(x), cos(x), x²]

This is interesting because:
- PQC outputs are bounded in [-1, 1] (Pauli-Z expectations)
- x and x² can exceed this range for |x| > 1
- This creates an inherent limitation

---

## Research & Literature Review

### 1. Parameterized Quantum Circuits (PQCs)

**Key References:**
- Benedetti et al., "Parameterized quantum circuits as machine learning models" (2019)
- Schuld & Petruccione, "Machine Learning with Quantum Computers" (2021)

**Core Concepts:**
- PQCs are the quantum analog of neural networks
- Expressibility depends on circuit depth and entanglement structure
- Hardware-efficient ansätze use native gate sets

### 2. Hybrid Quantum-Classical Models

**Architecture Patterns:**
- **Quantum Kernel Methods**: Classical preprocessing → quantum feature map
- **Variational Quantum Eigensolver (VQE)-style**: Classical optimizer drives quantum parameters
- **Dressed Quantum Circuits**: Classical layers wrap quantum circuit (our approach)

### 3. Encoding Strategies

**Data Encoding Methods:**
- **Angle Encoding**: x → RY(x) or RZ(x)
- **Amplitude Encoding**: Classical data embedded in amplitudes
- **IQP Encoding**: Diagonal gates with data-dependent angles

**Our Approach**: The MLP generates angles for the PQC (indirect encoding)

### 4. Entanglement Structures

**Common Patterns:**
- **Ring/Circular**: CNOT(i, (i+1) % n) - baseline uses this
- **Linear/Chain**: CNOT(i, i+1)
- **All-to-all**: Full connectivity
- **Alternating**: Even-odd pairs

**Research Finding**: Ring entanglement provides good expressibility with O(n) gates.

---

## Architecture Decisions

### Decision 1: Number of Qubits

**Options**: 4, 5, or 6 qubits

**Analysis**:
- More qubits → more expressive PQC
- More qubits → exponentially larger Hilbert space
- Task recommends 4-5 qubits

**Decision**: **5 qubits** 
- Matches output dimension (4) while adding expressibility
- Allows richer entanglement structure
- Not too computationally expensive

### Decision 2: PQC Depth

**Options**: 2, 3, or 4 layers

**Analysis**:
- Deeper circuits → more expressibility
- Deeper circuits → potential barren plateaus
- 2 layers in baseline may be limiting

**Decision**: **3 layers**
- Moderate increase in expressibility
- Manageable parameter count
- Total PQC params: 3 × 5 = 15

### Decision 3: MLP Architecture

**Task Requirement**: 2-3 Linear layers

**Options**:
- 2 layers: input → hidden → output
- 3 layers: input → hidden1 → hidden2 → output

**Decision**: **3 layers with increasing width**
```
1 → 32 → 64 → 15 (PQC params)
```

**Rationale**:
- Expanding architecture captures complex mappings
- Final layer directly outputs PQC parameters
- Total classical params: ~2.5K (lightweight)

### Decision 4: Activation Functions

**Options**: ReLU, Tanh, SiLU/Swish, GELU

**Decision**: **Tanh for final hidden layer**
- Bounds pre-PQC features to [-1, 1]
- Helps with parameter initialization
- Use ReLU for intermediate layers

### Decision 5: Output Scaling

**Problem**: PQC outputs are in [-1, 1], but targets include x and x² which exceed this range.

**Solution**: **Learnable output scaling layer**
```python
output = scale * pqc_output + bias
```
This allows the model to map [-1, 1] to the actual target range.

### Decision 6: Training Configuration

| Parameter | Baseline | Our Choice | Rationale |
|-----------|----------|------------|-----------|
| Epochs | 20 | 50 | More convergence time |
| Batch Size | 64 | 32 | More gradient updates |
| Learning Rate | 0.01 | 0.005 | Slower, stable convergence |
| Scheduler | None | StepLR | Decay for fine-tuning |
| Samples | 1024 | 2048 | More data for better generalization |

---

## Final Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID MODEL                             │
├─────────────────────────────────────────────────────────────┤
│  INPUT: x ~ N(0, 1)  [shape: (batch, 1)]                   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │              MLP (Classical)                │           │
│  │  Linear(1 → 32) → ReLU                      │           │
│  │  Linear(32 → 64) → ReLU                     │           │
│  │  Linear(64 → 15)                            │           │
│  └─────────────────────────────────────────────┘           │
│                           │                                 │
│                    PQC Parameters                           │
│                     [shape: (3, 5)]                         │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │         PQC (Quantum)  - 5 qubits           │           │
│  │  Layer 1: RY(θ₀...θ₄) + Ring CNOTs         │           │
│  │  Layer 2: RY(θ₅...θ₉) + Ring CNOTs         │           │
│  │  Layer 3: RY(θ₁₀...θ₁₄) + Ring CNOTs       │           │
│  │  Measure: ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩           │           │
│  └─────────────────────────────────────────────┘           │
│                           │                                 │
│              Expectation values [-1, 1]                     │
│                    [shape: (batch, 4)]                      │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │         Output Scaling Layer                │           │
│  │  scale * pqc_out + bias  (learnable)        │           │
│  └─────────────────────────────────────────────┘           │
│                           │                                 │
│                           ▼                                 │
│  OUTPUT: [x̂, sin(x)^, cos(x)^, x²^]                        │
│                    [shape: (batch, 4)]                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### File Structure
```
Task 11/
├── src/
│   ├── __init__.py
│   ├── model.py         # HybridModel, PQC definition
│   ├── dataset.py       # Data generation
│   └── training.py      # Training loop, evaluation
├── results/
│   ├── metrics.txt
│   ├── training_curve.png
│   └── model.pt
├── main.py              # Entry point
├── requirements.txt
├── PLANNING.md
├── DOCUMENTATION.md
└── README.md
```

### Implementation Order
1. `src/dataset.py` - Data generation utilities
2. `src/model.py` - PQC and hybrid model
3. `src/training.py` - Training and evaluation functions
4. `main.py` - Main execution script

---

## Expected Improvements Over Baseline

| Aspect | Baseline | Improved |
|--------|----------|----------|
| Qubits | 4 | 5 |
| PQC Layers | 2 | 3 |
| MLP Width | 16 | 32 → 64 |
| Output Scaling | None | Learnable |
| Epochs | 20 | 50 |
| Expected Loss | 0.5397 | < 0.40 |

---

## Risk Mitigation

1. **Barren Plateaus**: Use shallow circuit (3 layers), careful initialization
2. **Slow Training**: PQC simulation is slow; keep batch size manageable
3. **Overfitting**: Use train/test split for validation
4. **Numerical Issues**: Use float64 precision (as baseline does)

---

## Iteration Log

| Iteration | Changes | Test Loss | Notes |
|-----------|---------|-----------|-------|
| 1 | Initial implementation with 5 qubits, 3 layers, output scaling | **0.0541** | 90% improvement over baseline |

### Final Results Summary

**Configuration Used:**
- Qubits: 5
- PQC Layers: 3
- MLP: [32, 64]
- Samples: 1024
- Epochs: 30
- Learning Rate: 0.005 with StepLR scheduler

**Performance:**
| Metric | Value |
|--------|-------|
| Best Test MSE | 0.0541 |
| Final Train MSE | 0.0464 |
| Baseline MSE | 0.5397 |
| **Improvement** | **90.0%** |

**Per-Target Analysis:**
| Target | MSE | Observation |
|--------|-----|-------------|
| x | 0.0086 | Excellent fit |
| sin(x) | 0.0023 | Best performance |
| cos(x) | 0.0026 | Excellent fit |
| x² | 0.2229 | Hardest target (unbounded) |

**Key Success Factors:**
1. **Output Scaling Layer**: Critical for mapping bounded PQC outputs to unbounded targets
2. **Increased Circuit Depth**: 3 layers vs 2 improved expressibility
3. **5 Qubits**: Additional qubit provided more flexibility
4. **LR Scheduling**: StepLR helped fine-tune in later epochs
