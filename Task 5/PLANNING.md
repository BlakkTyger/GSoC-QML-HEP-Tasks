# Task V: Quantum Graph Neural Network (QGNN) - Planning Document

## Table of Contents
1. [Background and Context](#1-background-and-context)
2. [Understanding QGNNs: Theory and Concepts](#2-understanding-qgnns-theory-and-concepts)
3. [Literature Review: QGNN Architectures](#3-literature-review-qgnn-architectures)
4. [Implementation Frameworks: Cirq and TensorFlow Quantum](#4-implementation-frameworks-cirq-and-tensorflow-quantum)
5. [Iterative Architecture Selection](#5-iterative-architecture-selection)
6. [Final Architecture Design](#6-final-architecture-design)
7. [Implementation Plan](#7-implementation-plan)

---

## 1. Background and Context

### 1.1 Connection to Task II

In Task II, we implemented classical Graph Neural Networks (GNNs) for quark/gluon jet classification:
- **ParticleNet (EdgeConv)**: Dynamic k-NN graphs with edge convolutions
- **GAT (Graph Attention Network)**: Attention-weighted message passing

Key insights from Task II relevant to QGNN design:
- Jets are point clouds with ~15-50 particles (manageable for quantum circuits)
- Graph structure captures local particle correlations
- Features: (Δη, Δφ, log(pT), log(E), pT_frac)
- k-NN with k=16 provides good local connectivity

### 1.2 Motivation for Quantum GNNs

**Why explore QGNNs for jet classification?**

1. **Exponential Hilbert Space**: Quantum systems can represent correlations that scale exponentially with system size, potentially capturing complex multi-particle correlations
2. **Native Graph Structure**: Quantum circuits can naturally encode graph structure through entanglement topology
3. **Feature Map Expressiveness**: Quantum feature maps can embed classical data into high-dimensional spaces
4. **Potential Quantum Advantage**: Certain graph problems may benefit from quantum speedups

**Challenges:**
- Current NISQ devices have limited qubits and coherence
- Encoding classical graph data into quantum states is non-trivial
- Barren plateaus in variational circuits
- Measurement overhead

### 1.3 Task Requirements

**Primary Goal:** Design a QGNN circuit that:
1. Takes advantage of graph representation of jet data
2. Can be implemented and visualized using Cirq/TFQ
3. Performs classification (quark vs gluon)

---

## 2. Understanding QGNNs: Theory and Concepts

### 2.1 Classical GNN Recap

Classical GNN message passing:
```
h_v^(t+1) = UPDATE(h_v^(t), AGGREGATE({h_u^(t) : u ∈ N(v)}))
```

Key operations:
- **Aggregation**: Collect neighbor information
- **Update**: Transform node representations
- **Readout**: Global pooling for graph-level prediction

### 2.2 Quantum Analogues of GNN Operations

| Classical GNN | Quantum Analogue |
|---------------|------------------|
| Node features | Qubit states / Amplitude encoding |
| Edge connections | Entangling gates between qubits |
| Aggregation | Entanglement-based information mixing |
| Update | Parameterized single-qubit rotations |
| Readout | Measurement / Expectation values |

### 2.3 QGNN Core Principles

**Principle 1: Graph-Structured Entanglement**
- Map graph edges to entangling gates (CNOT, CZ, etc.)
- Nodes connected in the graph → qubits entangled in circuit
- Graph topology directly encoded in circuit structure

**Principle 2: Variational Quantum Circuits (VQC)**
- Parameterized gates trained via classical optimization
- Hybrid quantum-classical loop
- Gradient descent on expectation values

**Principle 3: Feature Encoding**
- Encode node features as rotation angles
- Data re-uploading: interleave encoding with variational layers
- Amplitude encoding for dense feature representation

### 2.4 Types of Quantum Graph Neural Networks

#### Type A: Quantum Walk-Based QGNN
- Based on continuous-time quantum walks on graphs
- Hamiltonian evolution: H = Σ_ij A_ij |i⟩⟨j|
- Evolution: |ψ(t)⟩ = e^{-iHt} |ψ(0)⟩

**Pros:** Direct physics interpretation, naturally captures graph structure
**Cons:** Requires Hamiltonian simulation, deep circuits

#### Type B: Variational QGNN with Graph-Structured Ansatz
- Parameterized quantum circuit with topology matching graph
- Layers: Encoding → Entanglement (follows edges) → Rotation → Repeat

**Pros:** Trainable, flexible, NISQ-friendly
**Cons:** May suffer from barren plateaus

#### Type C: Quantum Message Passing Neural Network (QMPNN)
- Quantum version of classical MPNN
- Messages computed via parameterized two-qubit gates
- Aggregation through entanglement

**Pros:** Direct analogue to successful classical architectures
**Cons:** Complex gate decomposition

#### Type D: Equivariant Quantum GNN
- Preserves graph symmetries (permutation equivariance)
- Uses symmetric quantum circuits

**Pros:** Better generalization
**Cons:** Restrictive ansatz

---

## 3. Literature Review: QGNN Architectures

### 3.1 Key Papers and Approaches

#### Paper 1: "A Quantum Graph Neural Network Approach to Particle Track Reconstruction"
**Authors:** Tüysüz et al. (2021)
**arXiv:** 2007.06868

**Key Ideas:**
- QGNN for particle track reconstruction at LHC
- Graph edges encoded as two-qubit interactions
- Variational circuit with graph-structured entanglement
- Used TensorFlow Quantum for implementation

**Architecture:**
```
Input Layer: Encode edge features as rotation angles
Hidden Layers: 
  - Single-qubit rotations (Rx, Ry, Rz)
  - Two-qubit entanglement following edge structure
Output: Edge classification (real edge vs fake)
```

**Results:** Competitive with classical methods on simplified tracking data

#### Paper 2: "Quantum Graph Neural Networks"
**Authors:** Verdon et al. (2019)
**arXiv:** 1909.12264

**Key Ideas:**
- Introduced quantum graph neural networks framework
- Graph structure encoded in Hamiltonian
- Trainable quantum walk parameters
- Applications to graph-level tasks

**Core Concept:**
```
H_QGNN = Σ_v θ_v Z_v + Σ_(u,v)∈E γ_{uv} (X_u X_v + Y_u Y_v)
```
- First term: Node-level rotations
- Second term: Edge-based entanglement (XY interaction)

#### Paper 3: "Hybrid Quantum-Classical Graph Neural Networks for Particle Physics"
**Authors:** Chen et al. (2022)

**Key Ideas:**
- Hybrid architecture: Classical preprocessing + Quantum core + Classical readout
- Reduces quantum resource requirements
- Feature dimensionality reduction before quantum encoding

**Architecture:**
```
Classical Encoder → Quantum Processing Unit → Classical Decoder
     MLP              Variational Circuit         MLP
```

#### Paper 4: "Quantum Machine Learning in High Energy Physics"
**Authors:** Guan et al. (2021)
**arXiv:** 2005.08582

**Key Ideas:**
- Survey of QML in HEP applications
- Discusses quantum classifiers for jet tagging
- Highlights challenges: encoding, training, measurement

#### Paper 5: "Graph Neural Networks with Quantum Attention"
**Authors:** Various

**Key Ideas:**
- Quantum implementation of attention mechanism
- Amplitude-based attention weights
- Potentially more expressive than classical attention

### 3.2 Implementation Patterns in Literature

| Pattern | Description | Used In |
|---------|-------------|---------|
| **Edge-based Entanglement** | Apply 2-qubit gates only where graph edges exist | Tüysüz et al., Verdon et al. |
| **Data Re-uploading** | Interleave data encoding with variational layers | Most VQC approaches |
| **Hybrid Layers** | Classical pre/post processing with quantum core | Chen et al. |
| **Hamiltonian Simulation** | Evolve under graph Laplacian | Verdon et al. |
| **Pooling via Measurement** | Measure subset of qubits for hierarchical structure | Hierarchical QGNN |

### 3.3 Relevance to Jet Classification

From the literature, key takeaways for our task:

1. **Graph structure matters**: Encoding edge topology in entanglement pattern is crucial
2. **Hybrid approaches work**: Don't need fully quantum - classical pre/post processing helps
3. **Feature encoding is critical**: Angle encoding most practical for NISQ
4. **Limited qubits**: Need to reduce particle count or use hierarchical approach
5. **HEP-specific work exists**: Can build on Tüysüz et al.'s approach

---

## 4. Implementation Frameworks: Cirq and TensorFlow Quantum

### 4.1 Google Cirq Overview

**Cirq** is Google's Python framework for designing, simulating, and running quantum circuits.

**Key Features:**
- Flexible qubit and gate definitions
- Native noise modeling
- Integration with TensorFlow Quantum
- Circuit visualization

**Basic QGNN Components in Cirq:**

```python
import cirq

# Define qubits (one per graph node)
qubits = cirq.LineQubit.range(n_nodes)

# Feature encoding (angle encoding)
def encode_features(qubits, features):
    for i, (q, f) in enumerate(zip(qubits, features)):
        yield cirq.rx(f[0])(q)  # Encode feature 0
        yield cirq.ry(f[1])(q)  # Encode feature 1

# Graph-structured entanglement
def graph_entanglement(qubits, edge_index, params):
    for idx, (i, j) in enumerate(edge_index.T):
        yield cirq.CNOT(qubits[i], qubits[j])
        yield cirq.rz(params[idx])(qubits[j])
        yield cirq.CNOT(qubits[i], qubits[j])

# Variational layer
def variational_layer(qubits, params):
    for i, q in enumerate(qubits):
        yield cirq.rx(params[3*i])(q)
        yield cirq.ry(params[3*i+1])(q)
        yield cirq.rz(params[3*i+2])(q)
```

### 4.2 TensorFlow Quantum (TFQ) Overview

**TFQ** extends TensorFlow to quantum machine learning, enabling:
- Hybrid quantum-classical models
- Automatic differentiation through quantum circuits
- Batch processing of quantum data
- Integration with Keras

**Key TFQ Components:**

```python
import tensorflow_quantum as tfq
import tensorflow as tf

# Convert Cirq circuit to TFQ layer
circuit = create_qgnn_circuit(qubits, edge_index)
circuit_tensor = tfq.convert_to_tensor([circuit])

# Create PQC layer
pqc_layer = tfq.layers.PQC(
    model_circuit,
    operators,  # Observables to measure
    differentiator=tfq.differentiators.Adjoint()
)

# Build Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    pqc_layer,
    tf.keras.layers.Dense(2, activation='softmax')
])
```

### 4.3 QGNN Architectures in TFQ

#### Architecture A: Simple Variational QGNN

```python
def create_variational_qgnn(qubits, edge_index, n_layers):
    circuit = cirq.Circuit()
    symbols = []
    
    for layer in range(n_layers):
        # Single-qubit rotations
        for i, q in enumerate(qubits):
            sym = sympy.Symbol(f'theta_{layer}_{i}')
            symbols.append(sym)
            circuit += cirq.ry(sym)(q)
        
        # Graph-structured entanglement
        for (i, j) in edges:
            circuit += cirq.CNOT(qubits[i], qubits[j])
    
    return circuit, symbols
```

#### Architecture B: Data Re-uploading QGNN

```python
def create_data_reuploading_qgnn(qubits, edge_index, features, n_layers):
    circuit = cirq.Circuit()
    
    for layer in range(n_layers):
        # Encode features (data re-uploading)
        for i, q in enumerate(qubits):
            circuit += cirq.rx(features[i][0])(q)
            circuit += cirq.ry(features[i][1])(q)
        
        # Variational rotations
        for i, q in enumerate(qubits):
            sym = sympy.Symbol(f'var_{layer}_{i}')
            circuit += cirq.rz(sym)(q)
        
        # Graph entanglement
        for (i, j) in edges:
            circuit += cirq.CZ(qubits[i], qubits[j])
    
    return circuit
```

#### Architecture C: Quantum Message Passing

```python
def quantum_message_passing(qubits, edge_index, params):
    circuit = cirq.Circuit()
    
    # Message computation (parameterized 2-qubit gates)
    for idx, (i, j) in enumerate(edges):
        # XX + YY interaction (exchange-type)
        circuit += cirq.XXPowGate(exponent=params[2*idx])(qubits[i], qubits[j])
        circuit += cirq.YYPowGate(exponent=params[2*idx+1])(qubits[i], qubits[j])
    
    # Update (single-qubit rotations)
    for i, q in enumerate(qubits):
        circuit += cirq.ry(params[len(edges)*2 + i])(q)
    
    return circuit
```

### 4.4 Practical Considerations

**Qubit Budget:**
- Current simulators: ~25-30 qubits practical
- Real devices: ~10-20 qubits with acceptable noise
- Jets have 15-50 particles → Need dimensionality reduction

**Strategies for Limited Qubits:**
1. **Subsampling**: Use top-k particles by pT
2. **Classical preprocessing**: Reduce to key features
3. **Hierarchical**: Process subgraphs sequentially
4. **Hybrid**: Use quantum only for core computation

**Encoding Strategies:**
- **Angle encoding**: 1 feature → 1 rotation, simple but limited capacity
- **Amplitude encoding**: n features → log(n) qubits, more compact but harder to prepare
- **Basis encoding**: Discrete features as computational basis states

---

## 5. Iterative Architecture Selection

### 5.1 Selection Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Graph Utilization** | High | How well does it leverage graph structure? |
| **NISQ Feasibility** | High | Practical on near-term devices/simulators? |
| **Expressiveness** | Medium | Can it learn complex patterns? |
| **Trainability** | High | Avoid barren plateaus, good gradients? |
| **Interpretability** | Medium | Can we understand what it learns? |
| **Implementation** | Medium | Feasible in Cirq/TFQ? |

### 5.2 Round 1: Candidate Architectures

Based on literature and framework capabilities:

| Architecture | Description |
|--------------|-------------|
| **A. Quantum Walk QGNN** | Hamiltonian evolution on graph Laplacian |
| **B. Variational Graph Ansatz** | Graph-structured entanglement with trainable rotations |
| **C. Data Re-uploading QGNN** | Interleaved encoding and variational layers |
| **D. Quantum Message Passing** | Parameterized 2-qubit gates as messages |
| **E. Hybrid Classical-Quantum GNN** | Classical encoder → Quantum core → Classical decoder |
| **F. Quantum Graph Attention** | Amplitude-based attention mechanism |

### 5.3 Round 2: Initial Filtering

| Architecture | Keep/Drop | Reason |
|--------------|-----------|--------|
| A. Quantum Walk | ✗ DROP | Requires deep circuits for Hamiltonian simulation |
| B. Variational Graph Ansatz | ✓ KEEP | Simple, NISQ-friendly, directly uses graph structure |
| C. Data Re-uploading | ✓ KEEP | Proven approach, good expressiveness |
| D. Quantum Message Passing | ✓ KEEP | Natural quantum analogue of classical GNN |
| E. Hybrid Classical-Quantum | ✓ KEEP | Practical, reduces quantum resources |
| F. Quantum Graph Attention | ✗ DROP | Complex, not well-established for NISQ |

**Remaining:** B, C, D, E

### 5.4 Round 3: Detailed Comparison

#### B. Variational Graph Ansatz

**Circuit Structure:**
```
|0⟩ ─[Ry(θ)]─●───────[Ry(θ)]─●───────
             │               │
|0⟩ ─[Ry(θ)]─X──●────[Ry(θ)]─X──●────
                │               │
|0⟩ ─[Ry(θ)]────X────[Ry(θ)]────X────
```

**Pros:**
- Direct graph structure encoding
- Simple to implement
- Low circuit depth

**Cons:**
- Limited expressiveness
- No data encoding in variational layers
- May underfit complex data

**Score: 7/10**

#### C. Data Re-uploading QGNN

**Circuit Structure:**
```
|0⟩ ─[Rx(x₁)][Ry(x₂)]─[Rz(θ)]─●───[Rx(x₁)][Ry(x₂)]─[Rz(θ)]─
                              │
|0⟩ ─[Rx(x₁)][Ry(x₂)]─[Rz(θ)]─X───[Rx(x₁)][Ry(x₂)]─[Rz(θ)]─
```

**Pros:**
- Feature encoding integrated
- Proven for classification tasks
- Universal function approximation

**Cons:**
- Deeper circuits
- Potential barren plateaus
- Graph structure less explicit

**Score: 8/10**

#### D. Quantum Message Passing

**Circuit Structure:**
```
Layer: [XX(θ)][YY(θ)] on edges → [Ry(θ)] on nodes → Repeat
```

**Pros:**
- Direct analogue to classical MPNN
- Physics-motivated (exchange interactions)
- Clear graph utilization

**Cons:**
- More complex gates (XX, YY decomposition)
- Higher gate count
- Harder to optimize

**Score: 7.5/10**

#### E. Hybrid Classical-Quantum GNN

**Architecture:**
```
Classical MLP → [Feature reduction to k dims] → Quantum Circuit → Classical MLP
```

**Pros:**
- Reduced quantum resources
- Classical parts handle heavy lifting
- Most practical for NISQ

**Cons:**
- Quantum part may be bottleneck
- Less "purely quantum"
- Quantum advantage unclear

**Score: 8.5/10**

### 5.5 Round 4: Final Selection

**Primary Architecture: Hybrid Data Re-uploading QGNN**

Combines the best aspects:
1. **Classical preprocessing**: Reduce particle count and feature dimension
2. **Data re-uploading**: Encode features multiple times for expressiveness
3. **Graph-structured entanglement**: Entangle based on k-NN edges
4. **Classical readout**: MLP for final classification

**Rationale:**
- Practical for current simulators (reduced qubit count)
- Good expressiveness through data re-uploading
- Explicit graph structure in entanglement pattern
- Proven in HEP applications (similar to Tüysüz et al.)

### 5.6 Design Decisions

**Node Selection:**
- Use top-8 particles by pT (8 qubits feasible for simulation)
- Captures dominant jet structure

**Feature Encoding:**
- 2-3 features per particle: (Δη, Δφ, log_pT_scaled)
- Angle encoding: Rx(Δη), Ry(Δφ), Rz(log_pT)

**Graph Construction:**
- k-NN with k=4 (reduced from k=16 for fewer edges)
- Results in ~8×4/2 = 16 unique edges (undirected)

**Circuit Layers:**
- 3 re-uploading layers
- Each: Encoding → Variational rotation → Graph entanglement

**Measurement:**
- Measure all qubits in Z basis
- Use parity or expectation values for classification

---

## 6. Final Architecture Design

### 6.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID QGNN ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Jet with N particles, features (η, φ, pT, E)           │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │     CLASSICAL PREPROCESSING                              │   │
│  │  1. Select top-8 particles by pT                        │   │
│  │  2. Compute relative coords: (Δη, Δφ)                   │   │
│  │  3. Normalize features to [-π, π]                       │   │
│  │  4. Build k-NN graph (k=4)                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │     QUANTUM CIRCUIT (8 qubits)                          │   │
│  │                                                          │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  LAYER 1-3 (repeated):                          │    │   │
│  │  │  ┌─────────────────────────────────────────┐    │    │   │
│  │  │  │ Feature Encoding:                        │    │    │   │
│  │  │  │   Rx(Δη_i) Ry(Δφ_i) Rz(log_pT_i)        │    │    │   │
│  │  │  └─────────────────────────────────────────┘    │    │   │
│  │  │                    ↓                            │    │   │
│  │  │  ┌─────────────────────────────────────────┐    │    │   │
│  │  │  │ Variational Rotation:                   │    │    │   │
│  │  │  │   Ry(θ_i) Rz(φ_i)  (trainable)         │    │    │   │
│  │  │  └─────────────────────────────────────────┘    │    │   │
│  │  │                    ↓                            │    │   │
│  │  │  ┌─────────────────────────────────────────┐    │    │   │
│  │  │  │ Graph Entanglement:                     │    │    │   │
│  │  │  │   CZ(q_i, q_j) for (i,j) ∈ Edges       │    │    │   │
│  │  │  └─────────────────────────────────────────┘    │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │                                                          │   │
│  │  Measurement: ⟨Z_0⟩, ⟨Z_1⟩, ..., ⟨Z_7⟩                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │     CLASSICAL READOUT                                   │   │
│  │  Dense(8 → 16) → ReLU → Dense(16 → 2) → Softmax        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  OUTPUT: [P(quark), P(gluon)]                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Quantum Circuit Detail

```
QGNN Circuit (3 layers, 8 qubits):

q0: ─[Rx(η₀)][Ry(φ₀)][Rz(pT₀)]─[Ry(θ₀)][Rz(ω₀)]─●───●───────────────[Layer 2...]─
                                                 │   │
q1: ─[Rx(η₁)][Ry(φ₁)][Rz(pT₁)]─[Ry(θ₁)][Rz(ω₁)]─Z───┼───●───────────[Layer 2...]─
                                                     │   │
q2: ─[Rx(η₂)][Ry(φ₂)][Rz(pT₂)]─[Ry(θ₂)][Rz(ω₂)]─────Z───┼───●───────[Layer 2...]─
                                                         │   │
q3: ─[Rx(η₃)][Ry(φ₃)][Rz(pT₃)]─[Ry(θ₃)][Rz(ω₃)]─────────Z───┼───●───[Layer 2...]─
                                                             │   │
q4: ─[Rx(η₄)][Ry(φ₄)][Rz(pT₄)]─[Ry(θ₄)][Rz(ω₄)]─────────────Z───┼───[Layer 2...]─
                                                                 │
q5: ─[Rx(η₅)][Ry(φ₅)][Rz(pT₅)]─[Ry(θ₅)][Rz(ω₅)]─────────────────Z───[Layer 2...]─
                                                                 
q6: ─[Rx(η₆)][Ry(φ₆)][Rz(pT₆)]─[Ry(θ₆)][Rz(ω₆)]───────────●─────────[Layer 2...]─
                                                           │
q7: ─[Rx(η₇)][Ry(φ₇)][Rz(pT₇)]─[Ry(θ₇)][Rz(ω₇)]───────────Z─────────[Layer 2...]─

(CZ gates shown only for subset of k-NN edges for clarity)
```

### 6.3 Parameter Count

| Component | Parameters |
|-----------|------------|
| Encoding | 0 (data-dependent) |
| Variational rotations | 8 qubits × 2 params × 3 layers = 48 |
| Classical readout | 8×16 + 16×2 = 160 |
| **Total** | **208 trainable parameters** |

### 6.4 Key Design Choices

**Why CZ for entanglement?**
- Native on many hardware platforms
- Creates entanglement without swapping amplitudes
- Symmetric (no directionality issues for undirected graphs)

**Why data re-uploading?**
- Single encoding layer limits expressiveness
- Re-uploading proven to achieve universal function approximation
- Allows quantum circuit to "see" data at different abstraction levels

**Why top-8 particles?**
- 8 qubits simulable on laptop
- Top particles by pT carry most jet information
- Quark/gluon differences most visible in leading particles

**Why k=4 for k-NN?**
- With 8 nodes, k=4 provides reasonable connectivity
- ~16 edges keeps circuit depth manageable
- Still captures local correlations

---

## 7. Implementation Plan

### 7.1 Code Structure

```
Task 5/
├── PLANNING.md                 # This document
├── DOCUMENTATION.md            # Final refined documentation
├── README.md                   # Task description
├── requirements.txt            # Dependencies
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Classical preprocessing, particle selection
│   ├── qgnn_circuit.py         # QGNN circuit definition in Cirq
│   ├── hybrid_model.py         # Full hybrid model with TFQ
│   └── visualization.py        # Circuit drawing and analysis
│
├── train.py                    # Training script
├── evaluate.py                 # Evaluation and comparison
└── results/                    # Saved models, circuit diagrams
```

### 7.2 Implementation Steps

1. **Data Preprocessing**
   - Load quark/gluon dataset (reuse from Task 2)
   - Select top-8 particles by pT
   - Compute relative coordinates and normalize
   - Build k-NN graph (k=4)

2. **Quantum Circuit (Cirq)**
   - Define 8 qubits
   - Implement encoding layer (Rx, Ry, Rz)
   - Implement variational layer (trainable Ry, Rz)
   - Implement graph entanglement (CZ on edges)
   - Stack into 3-layer circuit

3. **Hybrid Model (TFQ)**
   - Create input circuit with Sympy symbols
   - Define PQC layer with Z observables
   - Add classical Dense layers for readout
   - Compile with Adam optimizer, cross-entropy loss

4. **Training**
   - Batch processing of jets
   - Train on subset of data (quantum simulation is slow)
   - Track loss and accuracy

5. **Visualization**
   - Draw circuit diagram
   - Plot training curves
   - Visualize entanglement structure

### 7.3 Dependencies

```
cirq>=1.0.0
tensorflow>=2.10.0
tensorflow-quantum>=0.7.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
sympy>=1.10.0
energyflow>=1.3.0  # For dataset (optional, can reuse from Task 2)
```

### 7.4 Expected Outcomes

**Circuit Visualization:**
- Clear diagram showing graph-structured entanglement
- Layer-by-layer breakdown

**Performance:**
- Due to limited qubits and simulation overhead, don't expect to match classical GNN performance
- Goal: Demonstrate working QGNN concept
- Realistic AUC: ~0.60-0.70 (above random, showing learning)

**Insights:**
- How graph structure affects quantum circuit
- Comparison of quantum vs classical expressiveness
- Practical limitations of NISQ QGNN

---

## Research Sources

1. **QGNN for Particle Tracking:** C. Tüysüz et al., "A Quantum Graph Neural Network Approach to Particle Track Reconstruction", arXiv:2007.06868
2. **Original QGNN:** G. Verdon et al., "Quantum Graph Neural Networks", arXiv:1909.12264
3. **QML in HEP:** W. Guan et al., "Quantum Machine Learning in High Energy Physics", arXiv:2005.08582
4. **Data Re-uploading:** A. Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier", arXiv:1907.02085
5. **TensorFlow Quantum:** M. Broughton et al., "TensorFlow Quantum: A Software Framework for Quantum Machine Learning", arXiv:2003.02989
6. **Cirq Documentation:** https://quantumai.google/cirq
7. **VQC Training:** M. Cerezo et al., "Variational Quantum Algorithms", Nature Reviews Physics, 2021
8. **Barren Plateaus:** J. McClean et al., "Barren plateaus in quantum neural network training landscapes", Nature Communications, 2018
9. **From Task 2:** ParticleNet (arXiv:1902.08570), GAT (arXiv:1710.10903)

---

*Document created: Planning phase for Task V QGNN implementation*
*Next step: Implement the QGNN circuit and hybrid model*
