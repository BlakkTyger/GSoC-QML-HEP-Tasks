# QGAN for HEP Signal/Background Classification - Planning Document

## Table of Contents
1. [Dataset Analysis](#1-dataset-analysis)
2. [Physics Background](#2-physics-background)
3. [QGAN Fundamentals](#3-qgan-fundamentals)
4. [QGANs in HEP](#4-qgans-in-hep)
5. [Cirq/TFQ Implementation Research](#5-cirqtfq-implementation-research)
6. [Architecture Exploration](#6-architecture-exploration)
7. [Iterative Architecture Selection](#7-iterative-architecture-selection)
8. [Final Architecture Decision](#8-final-architecture-decision)
9. [Implementation Plan](#9-implementation-plan)
10. [References](#10-references)

---

## 1. Dataset Analysis

### 1.1 Data Structure
```
File: QIS_EXAM_200Events.npz
├── training_input (dict)
│   ├── '0' (background): shape=(50, 5), dtype=float64
│   └── '1' (signal): shape=(50, 5), dtype=float64
└── test_input (dict)
    ├── '0' (background): shape=(50, 5), dtype=float64
    └── '1' (signal): shape=(50, 5), dtype=float64
```

### 1.2 Key Statistics

| Metric | Training | Test |
|--------|----------|------|
| Total Samples | 100 | 100 |
| Signal (label=1) | 50 | 50 |
| Background (label=0) | 50 | 50 |
| Features | 5 | 5 |
| Feature Range | [-1, 1] | [-1, 1] |

### 1.3 Feature-wise Analysis (Training Data)

| Feature | Background Mean | Signal Mean | Separation |
|---------|-----------------|-------------|------------|
| 0 | -0.4546 | +0.3304 | **High** |
| 1 | -0.1039 | -0.2018 | Low |
| 2 | -0.2245 | -0.1568 | Low |
| 3 | -0.7970 | -0.6751 | Moderate |
| 4 | -0.5765 | -0.5401 | Low |

**Key Observation**: Feature 0 shows the strongest discriminative power between signal and background events.

### 1.4 Dataset Characteristics for ML
- **Small dataset**: Only 100 training samples → risk of overfitting
- **Balanced classes**: 50/50 split → no class imbalance issues
- **Pre-normalized**: Features already in [-1, 1] → suitable for quantum encoding
- **Low dimensionality**: 5 features → manageable qubit count

---

## 2. Physics Background

### 2.1 High Energy Physics Context

In High Energy Physics (HEP), particle collisions at accelerators like the LHC produce various types of events:

- **Signal Events**: The rare, interesting physics processes we want to study (e.g., Higgs boson production, new particle discovery)
- **Background Events**: Common Standard Model processes that mimic the signal signature

### 2.2 Delphes Simulation

The dataset was generated using **Delphes**, a fast multipurpose detector simulator that:
- Simulates detector response to particle collisions
- Includes realistic detector effects (resolution, efficiency)
- Used for phenomenology studies and analysis optimization

### 2.3 Common HEP Features

The 5 features likely represent kinematic variables such as:
- **Transverse momentum (pT)**: Momentum perpendicular to beam axis
- **Pseudorapidity (η)**: Angular distribution of particles
- **Azimuthal angle (φ)**: Angle in transverse plane
- **Invariant mass**: Reconstructed mass of particle systems
- **Missing transverse energy (MET)**: Signature of invisible particles (neutrinos, dark matter)

### 2.4 Classification Challenge

The goal is to build a classifier that can:
1. Learn the underlying distribution differences between signal and background
2. Accurately classify new events as signal or background
3. Maximize signal acceptance while minimizing background contamination

Traditional approaches: Boosted Decision Trees (BDT), Neural Networks (DNN)
Quantum approach: Leverage quantum feature spaces and interference effects

---

## 3. QGAN Fundamentals

### 3.1 Classical GAN Review

A Generative Adversarial Network consists of:
- **Generator (G)**: Learns to generate fake data that resembles real data
- **Discriminator (D)**: Learns to distinguish real data from fake data

Training is a minimax game:
```
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

### 3.2 Quantum GAN Variants

There are several QGAN architectures:

#### A. Full Quantum GAN
- Both Generator and Discriminator are quantum circuits
- Limited by current hardware capabilities
- Best for quantum data generation

#### B. Quantum Generator, Classical Discriminator (QCGAN)
- Generator: Parameterized Quantum Circuit (PQC)
- Discriminator: Classical Neural Network
- Most common hybrid approach

#### C. Classical Generator, Quantum Discriminator (CQGAN)
- Generator: Classical Neural Network
- Discriminator: Quantum Circuit
- Less common, useful for quantum-enhanced classification

#### D. Quantum Classifier with GAN-like Training
- Use quantum circuit as classifier
- Can employ adversarial training for robustness
- Most suitable for our classification task

### 3.3 Why QGANs for Classification?

**Important Clarification**: For signal/background classification, we don't need a traditional GAN structure. Instead, we have two approaches:

1. **Direct Quantum Classifier (QC)**: Use a PQC as a binary classifier
2. **QGAN for Data Augmentation**: Generate synthetic training data to improve classifier

Given our small dataset (100 samples), both approaches are worth exploring.

### 3.4 Quantum Advantages for HEP Classification

1. **Expressibility**: Quantum circuits can represent complex functions with fewer parameters
2. **Entanglement**: Can capture correlations between features that classical methods miss
3. **Quantum Feature Maps**: Transform data into high-dimensional Hilbert space
4. **Parameter Efficiency**: Fewer trainable parameters → less overfitting on small datasets

---

## 4. QGANs in HEP

### 4.1 Literature Review

#### Key Papers:

1. **"Quantum Machine Learning in High Energy Physics"** (Guan et al., 2021)
   - Survey of QML applications in HEP
   - Discusses VQC for event classification
   - Source: arXiv:2005.08582

2. **"Quantum Generative Adversarial Networks for HEP"** (Zoufal et al., 2019)
   - First QGAN implementation for loading probability distributions
   - Demonstrates quantum advantage in certain regimes
   - Source: arXiv:1904.00043

3. **"Application of Quantum Machine Learning to HEP Analysis"** (CERN, 2021)
   - Practical implementation for LHC data
   - Compares quantum vs classical classifiers
   - Source: arXiv:2103.12257

4. **"Anomaly Detection with QGANs"** (various, 2022)
   - Uses QGANs to identify anomalous events
   - Relevant for BSM physics searches
   - Source: Multiple arXiv papers

### 4.2 Successful Approaches in HEP

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| VQC Classifier | Simple, proven | Limited expressibility | Binary classification |
| Quantum Kernel | High accuracy | Computationally expensive | Small datasets |
| QGAN Augmentation | More training data | Complex training | Data scarcity |
| Hybrid Quantum-Classical | Best of both | Integration complexity | Production systems |

### 4.3 Relevance to Our Task

Given our constraints:
- Small dataset (100 samples)
- 5 features (5 qubits manageable)
- Binary classification
- Need to demonstrate fine-tuning

**Best approach**: Hybrid Quantum Classifier with systematic hyperparameter tuning

---

## 5. Cirq/TFQ Implementation Research

### 5.1 TensorFlow Quantum (TFQ) Overview

TFQ provides:
- Integration with TensorFlow/Keras ecosystem
- Cirq circuit construction
- Automatic differentiation for quantum circuits
- Batched quantum simulation

### 5.2 Key TFQ Components

```python
# Core imports
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy

# Key layers
tfq.layers.PQC           # Parameterized Quantum Circuit layer
tfq.layers.ControlledPQC # For more control
tfq.layers.Expectation   # Measure expectation values
tfq.layers.Sample        # Sample measurements
```

### 5.3 Circuit Design Patterns in TFQ

#### A. Data Encoding Circuits

1. **Angle Encoding**:
```python
def angle_encoding(qubits, x):
    for i, q in enumerate(qubits):
        cirq.ry(x[i] * np.pi)(q)
```

2. **Amplitude Encoding**:
   - Encodes data in amplitudes
   - More efficient but complex

3. **IQP Encoding**:
```python
def iqp_encoding(qubits, x):
    # Single-qubit rotations + ZZ interactions
    for i, q in enumerate(qubits):
        cirq.H(q)
        cirq.rz(x[i])(q)
    # Entangling layer
    for i in range(len(qubits)-1):
        cirq.ZZ(qubits[i], qubits[i+1])**(x[i]*x[i+1])
```

#### B. Variational Layers (Ansatz)

1. **Hardware Efficient Ansatz**:
```python
def hardware_efficient_layer(qubits, symbols):
    for i, q in enumerate(qubits):
        cirq.ry(symbols[2*i])(q)
        cirq.rz(symbols[2*i+1])(q)
    for i in range(len(qubits)-1):
        cirq.CNOT(qubits[i], qubits[i+1])
```

2. **Strongly Entangling Layer**:
```python
def strongly_entangling_layer(qubits, symbols):
    n = len(qubits)
    for i, q in enumerate(qubits):
        cirq.Rx(symbols[3*i])(q)
        cirq.Ry(symbols[3*i+1])(q)
        cirq.Rz(symbols[3*i+2])(q)
    for i in range(n):
        cirq.CNOT(qubits[i], qubits[(i+1) % n])
```

### 5.4 TFQ Training Pipeline

```python
# 1. Encode classical data to quantum circuits
x_train_circuits = [encode_data(x) for x in x_train]
x_train_tfq = tfq.convert_to_tensor(x_train_circuits)

# 2. Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tfq.layers.PQC(variational_circuit, readout_operators),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 3. Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_tfq, y_train, epochs=50, validation_split=0.2)
```

---

## 6. Architecture Exploration

### 6.1 Candidate Architectures

Based on research, I'll explore these architectures:

#### Architecture A: Simple VQC Classifier
```
Input (5 features) → Angle Encoding (5 qubits) → 
HW-Efficient Ansatz (2 layers) → Z Measurement → 
Classical Dense → Output
```
- **Pros**: Simple, fast training, interpretable
- **Cons**: May lack expressibility

#### Architecture B: Deep VQC with Re-uploading
```
[Encoding → Variational Layer] × 3 → 
Z Measurement → Classical Output
```
- **Pros**: Data re-uploading increases expressibility
- **Cons**: More parameters, slower training

#### Architecture C: Quantum Kernel Classifier
```
Quantum Feature Map → Kernel Matrix → 
Classical SVM/Logistic Regression
```
- **Pros**: Proven performance, no barren plateaus
- **Cons**: Scales poorly with data size

#### Architecture D: Hybrid GAN-style Classifier
```
Generator (optional augmentation) + 
Discriminator (quantum classifier)
```
- **Pros**: Addresses data scarcity
- **Cons**: Complex training dynamics

#### Architecture E: Entanglement-Enhanced Classifier
```
Dense Encoding → Full Entanglement Layer → 
Variational Ansatz → Multi-qubit Measurement
```
- **Pros**: Maximum quantum correlations
- **Cons**: May overfit small data

### 6.2 Initial Assessment Matrix

| Architecture | Expressibility | Trainability | Simplicity | HEP Suitability | Score |
|--------------|---------------|--------------|------------|-----------------|-------|
| A: Simple VQC | 3/5 | 5/5 | 5/5 | 4/5 | 17 |
| B: Deep VQC | 5/5 | 3/5 | 3/5 | 4/5 | 15 |
| C: Quantum Kernel | 4/5 | 5/5 | 4/5 | 5/5 | 18 |
| D: GAN-style | 4/5 | 2/5 | 2/5 | 3/5 | 11 |
| E: Entanglement | 5/5 | 3/5 | 3/5 | 4/5 | 15 |

---

## 7. Iterative Architecture Selection

### Iteration 1: Problem Requirements Analysis

**Constraints**:
- Small dataset (100 training samples)
- Binary classification
- Must demonstrate fine-tuning capability
- Use Cirq/TFQ
- Evaluate with Accuracy/AUC

**Requirements**:
- Avoid overfitting
- Interpretable hyperparameters
- Clear improvement metrics

**Decision**: Eliminate Architecture D (GAN-style) - too complex for the task and harder to demonstrate fine-tuning clearly.

**Remaining**: A, B, C, E

### Iteration 2: Overfitting Risk Assessment

With only 100 training samples, parameter count is critical:

| Architecture | Approx. Parameters | Overfitting Risk |
|--------------|-------------------|------------------|
| A: Simple VQC (2 layers) | 20 | Low |
| B: Deep VQC (3 re-uploads) | 45 | Medium |
| C: Quantum Kernel | 0 (kernel-based) | Very Low |
| E: Entanglement | 35 | Medium |

**Decision**: Keep all but prioritize low-parameter options.

### Iteration 3: Expressibility vs Data Size Trade-off

**Research Finding**: For small datasets, simpler models with regularization often outperform complex models.

**Analysis**:
- 100 samples, 5 features
- Rule of thumb: ~10 samples per parameter minimum
- Simple VQC (20 params) → 5 samples/param → marginal
- Deep VQC (45 params) → 2.2 samples/param → risky

**Decision**: Prioritize Architecture A and C. Keep B and E as secondary options with strong regularization.

### Iteration 4: Implementation Feasibility in TFQ

**Architecture A (Simple VQC)**:
- Direct TFQ implementation
- `tfq.layers.PQC` works out of box
- ✓ Feasible

**Architecture B (Deep VQC with Re-uploading)**:
- Requires custom circuit construction
- Need to interleave encoding and variational layers
- ✓ Feasible but more complex

**Architecture C (Quantum Kernel)**:
- TFQ doesn't have native kernel support
- Need to compute kernel matrix manually
- Then use sklearn SVM
- ✓ Feasible but hybrid approach

**Architecture E (Entanglement-Enhanced)**:
- Similar to A but with more entanglement
- ✓ Feasible

### Iteration 5: Fine-tuning Demonstration Capability

Key hyperparameters for fine-tuning:

| Architecture | Tunable Hyperparameters |
|--------------|------------------------|
| A: Simple VQC | num_layers, entanglement_type, learning_rate, encoding_type |
| B: Deep VQC | num_re-uploads, gates_per_layer, learning_rate |
| C: Quantum Kernel | feature_map_depth, kernel_type, SVM regularization |
| E: Entanglement | entanglement_pattern, num_layers, measurement_basis |

**Decision**: Architecture A offers the best balance of simplicity and tunability for demonstration.

### Iteration 6: Final Comparative Analysis

**Scoring Criteria** (1-5 scale):
1. Small data performance
2. Fine-tuning clarity
3. Implementation simplicity
4. Theoretical justification
5. Training stability

| Architecture | SD Perf | Fine-tune | Simple | Theory | Stable | **Total** |
|--------------|---------|-----------|--------|--------|--------|-----------|
| A: Simple VQC | 4 | 5 | 5 | 4 | 5 | **23** |
| B: Deep VQC | 3 | 4 | 3 | 5 | 3 | **18** |
| C: Quantum Kernel | 5 | 3 | 3 | 5 | 5 | **21** |
| E: Entanglement | 3 | 4 | 4 | 4 | 4 | **19** |

---

## 8. Final Architecture Decision

### 8.1 Primary Architecture: Enhanced VQC Classifier

Based on iterative analysis, I select a **modified Architecture A** with enhancements:

```
┌─────────────────────────────────────────────────────────────┐
│                    QGAN-Inspired VQC Classifier             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: x ∈ ℝ⁵ (5 HEP features)                            │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ DATA ENCODING LAYER                                  │   │
│  │ • Angle encoding: RY(π·xᵢ) on qubit i               │   │
│  │ • Optional: Add RZ(π·xᵢ) for richer encoding        │   │
│  └─────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ VARIATIONAL LAYER 1                                  │   │
│  │ • RY(θ) + RZ(φ) on each qubit                       │   │
│  │ • Entangling: CNOT ladder or circular              │   │
│  └─────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ VARIATIONAL LAYER 2 (optional, tunable)              │   │
│  │ • Same structure as Layer 1                         │   │
│  │ • Different parameters                              │   │
│  └─────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ MEASUREMENT                                          │   │
│  │ • Expectation value of Z₀ (or weighted sum of Zᵢ)   │   │
│  └─────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ CLASSICAL POST-PROCESSING                            │   │
│  │ • Linear layer: w·⟨Z⟩ + b                           │   │
│  │ • Sigmoid activation → probability                   │   │
│  └─────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  Output: P(signal) ∈ [0, 1]                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Architecture Specifications

| Component | Specification |
|-----------|---------------|
| Qubits | 5 (one per feature) |
| Encoding | Angle encoding (RY gates) |
| Variational Layers | 1-3 (tunable) |
| Gates per Layer | 10 (2 per qubit) + 4 CNOTs |
| Entanglement | Linear (CNOT ladder) |
| Measurement | ⟨Z₀⟩ expectation value |
| Total Parameters | 10-30 (depending on layers) |
| Loss Function | Binary Cross-Entropy |
| Optimizer | Adam |

### 8.3 QGAN Connection

While this is primarily a VQC classifier, it connects to QGAN concepts:
1. **Discriminator-like function**: The VQC acts like a discriminator, distinguishing signal from background
2. **Adversarial training potential**: Can add adversarial examples for robustness
3. **Generative extension**: Can add quantum generator for data augmentation (optional enhancement)

### 8.4 Fine-tuning Strategy

To demonstrate understanding of fine-tuning, I will systematically vary:

1. **Circuit Depth** (num_layers): 1, 2, 3
2. **Encoding Type**: RY only, RY+RZ
3. **Entanglement**: None, Linear, Circular, Full
4. **Learning Rate**: 0.01, 0.05, 0.1
5. **Batch Size**: 10, 25, 50

### 8.5 Evaluation Metrics

- **Primary**: Classification Accuracy
- **Secondary**: AUC-ROC
- **Additional**: Precision, Recall, F1-Score

---

## 9. Implementation Plan

### 9.1 Code Structure

```
Task 4/
├── analyze_data.py          # Data exploration (done)
├── QGAN_PLANNING_DOCUMENT.md # This document
├── qgan_classifier.py       # Main implementation
│   ├── DataLoader           # Load and preprocess data
│   ├── QuantumCircuits      # Circuit builders
│   ├── QGANClassifier       # Main model class
│   └── Trainer              # Training loop
├── hyperparameter_tuning.py # Systematic tuning
├── results/                 # Output directory
│   ├── training_history.json
│   ├── model_comparison.png
│   └── best_model.h5
└── FINAL_DOCUMENTATION.md   # Refined final report
```

### 9.2 Implementation Steps

1. **Data Preparation**
   - Load NPZ file
   - Create train/validation split
   - Convert to TFQ tensor format

2. **Circuit Construction**
   - Implement encoding functions
   - Implement variational layers
   - Create parameterized measurement

3. **Model Building**
   - Use tf.keras.Sequential
   - Integrate TFQ layers
   - Add classical post-processing

4. **Training Pipeline**
   - Configure optimizer
   - Set up callbacks (early stopping, checkpoints)
   - Train with validation

5. **Evaluation**
   - Compute accuracy and AUC
   - Generate ROC curves
   - Compare configurations

6. **Hyperparameter Tuning**
   - Grid search over key parameters
   - Record all results
   - Select best configuration

7. **Documentation**
   - Summarize findings
   - Create visualizations
   - Write final report

### 9.3 Expected Timeline

| Phase | Estimated Effort |
|-------|-----------------|
| Data Preparation | 10% |
| Circuit Construction | 20% |
| Model Building | 20% |
| Training & Evaluation | 30% |
| Fine-tuning | 15% |
| Documentation | 5% |

---

## 10. References

### Academic Papers
1. Zoufal, C., Lucchi, A., & Woerner, S. (2019). Quantum Generative Adversarial Networks for Learning and Loading Random Distributions. *npj Quantum Information*, 5(1), 1-9. [arXiv:1904.00043]

2. Guan, W., et al. (2021). Quantum Machine Learning in High Energy Physics. *Machine Learning: Science and Technology*, 2(1), 011003. [arXiv:2005.08582]

3. Schuld, M., & Petruccione, F. (2018). Supervised Learning with Quantum Computers. *Springer*.

4. Havlíček, V., et al. (2019). Supervised Learning with Quantum-Enhanced Feature Spaces. *Nature*, 567(7747), 209-212.

5. Benedetti, M., et al. (2019). Parameterized Quantum Circuits as Machine Learning Models. *Quantum Science and Technology*, 4(4), 043001.

### Documentation & Tutorials
- TensorFlow Quantum Documentation: https://www.tensorflow.org/quantum
- Cirq Documentation: https://quantumai.google/cirq
- TFQ Tutorials: https://www.tensorflow.org/quantum/tutorials
- Pennylane QML Demos: https://pennylane.ai/qml/demos.html

### HEP Resources
- CERN Open Data Portal: http://opendata.cern.ch/
- Delphes Framework: https://cp3.irmp.ucl.ac.be/projects/delphes

---

## Appendix: Key Decisions Log

| Decision Point | Options Considered | Choice | Reasoning |
|---------------|-------------------|--------|-----------|
| Task Interpretation | Full QGAN vs VQC Classifier | VQC Classifier with QGAN concepts | Better suited for classification task |
| Architecture | 5 candidates | Enhanced Simple VQC | Best balance of simplicity, trainability, fine-tuning |
| Encoding | Amplitude, Angle, IQP | Angle (RY) | Simple, effective for small data |
| Entanglement | None, Linear, Full | Linear (tunable) | Good balance |
| Measurement | Single qubit, Multi-qubit | Single (Z₀) | Simplest, can extend |
| Framework | Pure Cirq, TFQ, Pennylane | TFQ | Required by task, good Keras integration |

---

*Document created: March 2026*
*Author: GSoC-QML-HEP Task 4*
*Status: Planning Complete - Ready for Implementation*
