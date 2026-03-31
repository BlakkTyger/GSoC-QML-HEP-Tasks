# Task 7: Z₂ × Z₂ Equivariant Quantum Neural Networks - Planning Document

## 1. Problem Statement

Implement a Z₂ × Z₂ equivariant quantum neural network for a binary classification task on a 2D dataset that respects Z₂ × Z₂ symmetry (Klein four-group). Compare performance against a standard (non-equivariant) QNN.

### Key Objectives
- Generate a synthetic classification dataset with Z₂ × Z₂ symmetry
- Implement a standard QNN as baseline
- Implement a Z₂ × Z₂ equivariant QNN
- Compare generalization performance, training efficiency, and parameter count

---

## 2. Theoretical Background & Research

### 2.1 The Z₂ × Z₂ Symmetry Group (Klein Four-Group)

The Klein four-group V₄ = Z₂ × Z₂ is the smallest non-cyclic group, consisting of four elements:

```
V₄ = {(0,0), (0,1), (1,0), (1,1)}
```

With group operation being entry-wise addition modulo 2.

**Physical Interpretation for 2D Data (x₁, x₂):**
- **(0,0)**: Identity - no transformation
- **(1,0)**: Exchange coordinates: (x₁, x₂) → (x₂, x₁) (reflection along y=x)
- **(0,1)**: Sign flip: (x₁, x₂) → (-x₁, -x₂) (point reflection through origin)
- **(1,1)**: Combined: (x₁, x₂) → (-x₂, -x₁) (reflection along y=-x)

**Source**: Meyer et al., "Exploiting symmetry in variational quantum machine learning" (arXiv:2205.06217)

### 2.2 Representation Theory for Equivariant QNNs

**Key Concept: Equivariance**

A function f is equivariant with respect to symmetry group S if:
```
f(V_s · x) = U_s · f(x)  ∀s ∈ S
```

Where:
- V_s is the representation acting on input data
- U_s is the representation acting on the output/Hilbert space

**For Invariant Predictions:**
The prediction must satisfy: `pred(V_s · x) = pred(x)` for all s ∈ S

**Source**: Nguyen et al., "Theory for Equivariant Quantum Neural Networks" (arXiv:2210.08566)

### 2.3 Data Representation on Hilbert Space

For the Z₂ × Z₂ symmetry with 2D data embedded via angle encoding:

**Data Embedding**: U(x) = R_Z(x₁) ⊗ R_Z(x₂)

**Induced Hilbert Space Representations**:
```
U_(0,0) = I ⊗ I           (Identity)
U_(1,0) = SWAP            (Exchange qubits)
U_(0,1) = X ⊗ X           (Sign flip via Pauli-X)
U_(1,1) = SWAP · (X ⊗ X)  (Combined operation)
```

This follows from the property: X·Z·X = -Z, which means R_Z(-θ) = X · R_Z(θ) · X

**Source**: Meyer et al. (2205.06217), Section II

### 2.4 Gate Symmetrization via Twirling

**The Twirling Formula:**
```
T_U[G] = (1/|S|) Σ_{s∈S} U_s† · G · U_s
```

This projects any operator G onto the commutant of the representation, i.e., operators that commute with all U_s.

**Proposition**: A gate R_G[θ] = exp(-iθG) is equivariant iff [G, U_s] = 0 for all s ∈ S.

**Source**: Meyer et al. (2205.06217), Proposition 1-2

### 2.5 Z₂ × Z₂ Equivariant Gateset Derivation

Starting from standard gateset: {X₁, Y₁, Z₁, X₂, Y₂, Z₂, ZZ}

**Step 1: Symmetrize w.r.t. Exchange (SWAP)**
- Local gates become simultaneous: X₁ → (X₁ + X₂)/2 → X⊗I + I⊗X
- ZZ already commutes with SWAP

**Step 2: Symmetrize w.r.t. Sign Flip (X⊗X)**
- X gates commute with X⊗X: [X₁, X₁X₂] = 0 ✓
- Y, Z gates anti-commute: X·Y·X = -Y, X·Z·X = -Z
- Therefore Y and Z gates are eliminated!

**Final Equivariant Gateset for Z₂ × Z₂:**
```
G_equiv = {X₁ + X₂, Z₁Z₂}
```

This dramatically reduces the parameter space while preserving symmetry.

**Source**: Meyer et al. (2205.06217), Section III.B

---

## 3. Dataset Design

### 3.1 Z₂ × Z₂ Symmetric Dataset Requirements

The dataset must satisfy:
```
label(x₁, x₂) = label(x₂, x₁) = label(-x₁, -x₂) = label(-x₂, -x₁)
```

### 3.2 Proposed Dataset: Symmetric Checkerboard/Quadrant Pattern

**Design Rationale:**
- Two features x₁, x₂ ∈ [-1, 1]
- Classification based on sign patterns that respect all symmetries
- Class assignment: Based on |x₁| + |x₂| vs threshold, or quadrant-based rules

**Concrete Dataset Construction:**

**Option A: Radial Symmetric Dataset**
```python
# Class 0: Points where |x₁·x₂| < threshold AND some radial condition
# Class 1: Otherwise
# This naturally respects all Z₂ × Z₂ transformations
```

**Option B: Product-Based Classification (Recommended)**
```python
# Generate points in [-1, 1] × [-1, 1]
# Label based on: sign(x₁ · x₂) or |x₁ · x₂| > threshold
# Product x₁·x₂ is invariant under all Z₂ × Z₂ transformations:
#   - x₁·x₂ = x₂·x₁ ✓
#   - (-x₁)·(-x₂) = x₁·x₂ ✓
#   - (-x₂)·(-x₁) = x₁·x₂ ✓
```

**Option C: Distance-Based with Symmetric Centers**
```python
# Place cluster centers at symmetric positions
# e.g., centers at (a, a), (-a, -a), (a, -a), (-a, a)
# Classification by nearest center group
```

**Selected Approach**: Option B (Product-based) for mathematical elegance and clear symmetry preservation.

### 3.3 Dataset Parameters
- Training samples: 200-500
- Test samples: 100-200
- Feature range: [-1, 1] × [-1, 1]
- Class balance: ~50/50
- Add controlled noise for realistic evaluation

---

## 4. Architecture Analysis & Selection

### 4.1 Candidate Architectures

#### Architecture 1: Standard VQC (Non-Equivariant Baseline)

**Structure:**
```
|0⟩ --[Embedding]--[Variational Layer]^L--[Measurement]
```

**Embedding**: Angle encoding R_Z(x₁) ⊗ R_Z(x₂)

**Variational Layer**:
- Single-qubit rotations: R_X(θ), R_Y(θ), R_Z(θ) on each qubit
- Entangling: CNOT or CZ gates
- Parameters per layer: 6 (3 rotations × 2 qubits)

**Pros:**
- Full expressivity
- Standard implementation

**Cons:**
- No inductive bias
- Prone to overfitting
- More parameters than necessary

#### Architecture 2: Z₂ × Z₂ Equivariant QNN (Gate Symmetrization)

**Structure:**
```
|0⟩ --[Equiv. Embedding]--[Equiv. Variational Layer]^L--[Invariant Measurement]
```

**Equivariant Embedding**: Same angle encoding (naturally equivariant)

**Equivariant Variational Layer**:
- Only use gates from G_equiv = {X₁ + X₂, Z₁Z₂}
- R_XX(θ) = exp(-iθ(X⊗I + I⊗X)/2) — simultaneous X rotation
- R_ZZ(θ) = exp(-iθZ⊗Z) — entangling ZZ rotation
- Parameters per layer: 2

**Invariant Initial State**: |00⟩ (invariant under all symmetries)

**Invariant Observable**: Z⊗Z (commutes with all U_s)

**Pros:**
- Built-in inductive bias
- Fewer parameters
- Better generalization
- Barren plateau mitigation

**Cons:**
- Reduced expressivity (by design)
- May underfit if symmetry is only approximate

#### Architecture 3: Hybrid Partial-Equivariant

**Concept**: Use equivariant layers with occasional symmetry-breaking gates

**Rationale**: Fine-tune trade-off between equivariance and expressivity

**Structure:**
- Primarily equivariant layers
- Add limited non-equivariant gates (e.g., single R_Y or R_Z)

**When to use**: If pure equivariant underperforms due to insufficient expressivity

#### Architecture 4: Data Re-uploading Equivariant Model

**Structure:**
```
|0⟩ --[W₁]--[U(x)]--[W₂]--[U(x)]--...--[Measurement]
```

**Concept**: Interleave equivariant trainable blocks with data encoding

**Advantages:**
- Enhanced expressivity through re-uploading
- Maintains equivariance throughout

**Source**: Meyer et al. (2205.06217), Section IV

### 4.2 Architecture Comparison Matrix

| Criterion | Standard VQC | Equiv. QNN | Hybrid | Re-uploading |
|-----------|--------------|------------|--------|--------------|
| Parameters | High (6L) | Low (2L) | Medium | Medium-High |
| Symmetry | None | Full Z₂×Z₂ | Partial | Full Z₂×Z₂ |
| Expressivity | Full | Constrained | Balanced | High |
| Generalization | Poor expected | Good expected | Moderate | Good |
| Barren Plateaus | Higher risk | Lower risk | Moderate | Lower risk |
| Implementation | Simple | Moderate | Complex | Moderate |

### 4.3 Selected Architectures

**Primary Implementation:**
1. **Standard VQC** (Architecture 1) — Baseline for comparison
2. **Z₂ × Z₂ Equivariant QNN** (Architecture 2) — Main equivariant model

**Secondary (if time permits):**
3. **Data Re-uploading Equivariant** (Architecture 4) — For enhanced performance

---

## 5. Implementation Plan

### 5.1 Project Structure

```
Task 7/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Z₂×Z₂ symmetric dataset generation
│   ├── embeddings.py       # Quantum data encodings
│   ├── standard_qnn.py     # Non-equivariant QNN
│   ├── equivariant_qnn.py  # Z₂×Z₂ equivariant QNN
│   ├── training.py         # Training loops & optimization
│   └── utils.py            # Visualization, metrics
├── results/
│   ├── figures/
│   └── metrics/
├── main.py                 # Main execution script
├── requirements.txt
├── PLANNING.md
├── DOCUMENTATION.md
└── README.md
```

### 5.2 Implementation Steps

**Phase 1: Foundation**
1. Set up project structure
2. Implement dataset generation with Z₂ × Z₂ symmetry
3. Create visualization tools for dataset and symmetry verification

**Phase 2: Standard QNN Baseline**
4. Implement angle encoding
5. Build standard variational circuit
6. Implement training loop with cost function
7. Train and evaluate baseline

**Phase 3: Equivariant QNN**
8. Implement equivariant gates (R_XX, R_ZZ)
9. Build equivariant variational circuit
10. Use invariant observable (Z⊗Z)
11. Train and evaluate equivariant model

**Phase 4: Comparison & Analysis**
12. Compare training curves
13. Compare generalization (train vs test accuracy)
14. Compare parameter efficiency
15. Visualize decision boundaries

### 5.3 Technical Specifications

**Framework**: PennyLane (preferred for equivariant QML support)

**Optimizer**: Adam with learning rate scheduling

**Cost Function**: 
- MSE or Cross-entropy loss
- Optionally add symmetry penalty term for baseline

**Metrics**:
- Training/Test accuracy
- Generalization gap
- Convergence speed
- Parameter count

**Hardware**: Default qubit simulator (no noise model initially)

---

## 6. Expected Outcomes & Hypotheses

### 6.1 Hypotheses

**H1**: The equivariant QNN will achieve better test accuracy than the standard QNN with fewer training samples.

**H2**: The equivariant QNN will have a smaller generalization gap (difference between train and test accuracy).

**H3**: The equivariant QNN will converge faster due to the reduced parameter space.

**H4**: The standard QNN will achieve higher training accuracy but lower test accuracy (overfitting).

### 6.2 Expected Results

| Metric | Standard QNN | Equivariant QNN |
|--------|--------------|-----------------|
| Train Accuracy | ~95-100% | ~85-95% |
| Test Accuracy | ~70-85% | ~85-95% |
| Parameters | 12-24 | 4-8 |
| Convergence | Slower | Faster |
| Gen. Gap | Large | Small |

---

## 7. Research Sources & References

### 7.1 Primary References

1. **Meyer, J.J., et al.** (2023). "Exploiting symmetry in variational quantum machine learning." *PRX Quantum*, 4, 010328. [arXiv:2205.06217](https://arxiv.org/abs/2205.06217)
   - Core methodology for gate symmetrization and Z₂ × Z₂ example
   - Twirling formula for equivariant gateset construction

2. **Nguyen, Q.T., et al.** (2024). "Theory for Equivariant Quantum Neural Networks." *PRX Quantum*, 5, 020328. [arXiv:2210.08566](https://arxiv.org/abs/2210.08566)
   - Comprehensive EQNN theory
   - Lie group representations
   - Barren plateau mitigation through symmetry

### 7.2 Supporting References

3. **Larocca, M., et al.** (2022). "Group-Invariant Quantum Machine Learning." *PRX Quantum*, 3, 030341.
   - Theoretical foundations of invariant QML

4. **Bronstein, M.M., et al.** (2021). "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
   - Classical geometric deep learning blueprint

5. **PennyLane Documentation**: "Introduction to Geometric Quantum Machine Learning"
   - Practical implementation guidance
   - Twirling examples

### 7.3 Additional Background

6. **Skolik, A., et al.** (2023). "Equivariant quantum circuits for learning on weighted graphs." *npj Quantum Information*.

7. **Schatzki, L., et al.** (2022). "Theoretical guarantees for permutation-equivariant quantum neural networks."

---

## 8. Risk Assessment & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Equivariant gateset too restrictive | Model underfits | Add limited symmetry-breaking gates |
| Barren plateaus in standard QNN | Training fails | Use small initialization, layer-wise training |
| Dataset too simple | Trivial classification | Increase complexity, add noise |
| PennyLane compatibility | Implementation delays | Fall back to Qiskit if needed |

---

## 9. Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| 1 | Foundation + Dataset | 1-2 hours |
| 2 | Standard QNN | 2-3 hours |
| 3 | Equivariant QNN | 2-3 hours |
| 4 | Analysis + Documentation | 2 hours |

**Total Estimated Time**: 7-10 hours

---

## 10. Success Criteria

1. ✓ Dataset correctly implements Z₂ × Z₂ symmetry (verified visually and mathematically)
2. ✓ Both QNN models train successfully
3. ✓ Equivariant QNN shows improved generalization over standard QNN
4. ✓ Clear comparison with metrics and visualizations
5. ✓ Clean, modular, reproducible code
6. ✓ Comprehensive documentation

---

*Document Version: 1.0*
*Created: March 2025*
*Author: GSoC-QML-HEP Project*
