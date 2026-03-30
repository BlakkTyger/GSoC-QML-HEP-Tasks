# Task II: Classical GNN for Quark/Gluon Classification - Planning Document

## Table of Contents
1. [Understanding the Dataset and Physics](#1-understanding-the-dataset-and-physics)
2. [Graph Neural Networks in HEP](#2-graph-neural-networks-in-hep)
3. [Graph Construction Strategies](#3-graph-construction-strategies)
4. [GNN Architecture Research](#4-gnn-architecture-research)
5. [Iterative Architecture Selection](#5-iterative-architecture-selection)
6. [Final Architecture Decisions](#6-final-architecture-decisions)
7. [Implementation Plan](#7-implementation-plan)

---

## 1. Understanding the Dataset and Physics

### 1.1 Physics Background: Quark vs Gluon Jets

**Jets** are collimated sprays of hadrons produced when quarks or gluons are created in high-energy particle collisions. Due to **color confinement** in QCD (Quantum Chromodynamics), quarks and gluons cannot exist freely and instead "hadronize" into jets of observable particles.

**Why distinguish quark and gluon jets?**
- Gluon jets tend to have **higher particle multiplicity** (more particles) due to gluons carrying more color charge
- Gluon jets are typically **broader** (larger angular spread)
- Quark jets tend to be more **collimated** with harder fragmentation
- This classification is crucial for:
  - Reducing QCD backgrounds in Higgs and BSM searches
  - Improving precision measurements
  - Understanding jet substructure

**Physical observables that distinguish q/g jets:**
| Property | Quark Jets | Gluon Jets |
|----------|------------|------------|
| Multiplicity | Lower (~15-20 particles) | Higher (~20-30 particles) |
| pT distribution | Harder fragmentation | Softer fragmentation |
| Jet width | Narrower | Wider |
| Color factor | CF = 4/3 | CA = 3 |

### 1.2 Dataset Structure

**Source:** Zenodo dataset from ParticleNet paper, accessible via EnergyFlow package

**Data Format:**
```
X: (N_jets, M, 4) - Particle-level features
   - N_jets = 100,000 per file (50k quark, 50k gluon)
   - M = max multiplicity (variable padding with zeros)
   - 4 features per particle: (pT, rapidity, azimuthal angle φ, pdgid)

y: (N_jets,) - Labels
   - 0 = gluon
   - 1 = quark
```

**Key Dataset Characteristics:**
- **Point cloud structure**: Each jet is an unordered set of particles (permutation invariant)
- **Variable size**: Jets have different numbers of particles (need handling)
- **Kinematic features**: pT, η (rapidity), φ provide 3D momentum space representation
- **Particle ID**: pdgid encodes particle type (can be used or ignored)

**Kinematic Cuts Applied:**
- pT_jet ∈ [500, 550] GeV (narrow pT window reduces pT-dependent effects)
- |y_jet| < 1.7 (central jets)
- anti-kT algorithm with R=0.4

### 1.3 Why Graphs for Jet Data?

Jets are naturally represented as **point clouds** - unordered sets of particles in (η, φ) space. Graphs provide:

1. **Permutation invariance**: No arbitrary ordering imposed on particles
2. **Relational learning**: Capture correlations between particle pairs
3. **Variable size handling**: Naturally handle different numbers of constituents
4. **Physics-motivated structure**: Can encode physically meaningful relationships

---

## 2. Graph Neural Networks in HEP

### 2.1 Literature Review

**Key Papers in GNN-based Jet Classification:**

1. **ParticleNet (2019)** - Qu & Gouskos
   - arXiv: 1902.08570
   - Introduced EdgeConv for jets, dynamic graph construction
   - State-of-the-art performance on quark/gluon classification
   - Key idea: k-NN graph in learned feature space, updated dynamically

2. **Energy Flow Networks (2018)** - Komiske, Metodiev, Thaler
   - arXiv: 1810.05165
   - Deep Sets architecture for IRC-safe jet classification
   - Established baseline for this dataset

3. **Jet Tagging via Particle Clouds (2020)** - Mikuni & Canelli
   - arXiv: 2002.02967
   - Compared various architectures including attention mechanisms

4. **LorentzNet (2022)** - Gong et al.
   - arXiv: 2201.08187
   - Lorentz-equivariant GNN respecting spacetime symmetries

5. **Graph Attention Networks for Particle Physics (2020)**
   - Various works applying GAT to jet classification
   - Attention provides interpretable particle importance

6. **Message Passing Neural Networks for Jet Physics**
   - General MPNN framework applied to particle physics

### 2.2 GNN Paradigms Relevant to Jets

| Architecture | Key Mechanism | Pros | Cons |
|--------------|---------------|------|------|
| **GCN** | Spectral convolution | Simple, efficient | Requires fixed graph |
| **GraphSAGE** | Neighborhood sampling | Scalable, inductive | Less expressive |
| **GAT** | Attention weights | Interpretable, adaptive | Computationally heavier |
| **EdgeConv/DGCNN** | Dynamic edge features | Learns graph structure | Requires k-NN recomputation |
| **GIN** | Sum aggregation | Maximally expressive | May overfit |
| **TransformerConv** | Full attention | Global context | O(n²) complexity |

---

## 3. Graph Construction Strategies

### 3.1 The Core Challenge

Converting a point cloud of particles into a graph requires defining:
1. **Nodes**: What entity does each node represent?
2. **Edges**: How do we connect nodes?
3. **Node features**: What information does each node carry?
4. **Edge features**: What information does each edge carry?

### 3.2 Graph Construction Options

#### Option A: k-Nearest Neighbors (k-NN) Graph
```
For each particle i:
    Find k nearest particles in (η, φ) space
    Create edges (i, j) for each neighbor j
```
**Pros:**
- Captures local structure
- Fixed number of edges per node (good for batching)
- Physics-motivated: nearby particles are correlated

**Cons:**
- Choice of k is a hyperparameter
- May miss long-range correlations

**Distance Metric Options:**
- Euclidean: d = √[(Δη)² + (Δφ)²]
- Angular: ΔR = √[(Δη)² + (Δφ)²] (same as Euclidean in η-φ space)
- pT-weighted: d_ij = min(pT_i, pT_j) × ΔR_ij (physics-motivated)

#### Option B: Fully Connected Graph
```
Connect every particle to every other particle
```
**Pros:**
- No information loss
- Can learn any structure

**Cons:**
- O(n²) edges - expensive for high multiplicity jets
- May learn spurious correlations

#### Option C: Radius Graph (ε-ball)
```
Connect particles within distance ε in (η, φ) space
```
**Pros:**
- Adaptive connectivity based on local density

**Cons:**
- Variable number of edges (batching challenges)
- Isolated nodes possible

#### Option D: Dynamic Graph (EdgeConv-style)
```
Initial: k-NN in (η, φ) space
After each layer: k-NN in learned feature space
```
**Pros:**
- Learns optimal graph structure
- State-of-the-art results (ParticleNet)

**Cons:**
- Computationally expensive (k-NN at each layer)

### 3.3 Feature Engineering

**Node Features (per particle):**
| Feature | Description | Normalization |
|---------|-------------|---------------|
| pT | Transverse momentum | log(pT) or pT/pT_jet |
| Δη | Rapidity relative to jet axis | Raw or standardized |
| Δφ | Azimuthal angle relative to jet axis | Raw or standardized |
| E | Energy | log(E) or E/E_jet |
| pdgid | Particle type | One-hot or embedding |

**Derived Features (physics-motivated):**
- log(pT): Compresses dynamic range
- log(E): Energy on log scale
- pT/pT_jet: Relative momentum fraction
- ΔR = √(Δη² + Δφ²): Angular distance from jet axis

**Edge Features (per edge i→j):**
| Feature | Description |
|---------|-------------|
| ΔR_ij | Angular distance |
| Δη_ij, Δφ_ij | Component distances |
| k_T,ij = min(pT_i, pT_j) × ΔR_ij | Generalized kT distance |
| m_ij | Invariant mass of pair |

### 3.4 Decision: Graph Construction Approach

**Primary Choice: k-NN Graph with k=16**

**Reasoning:**
1. Follows ParticleNet's proven approach
2. Captures local correlations which are physically meaningful
3. Fixed edge count enables efficient batching
4. k=16 provides good balance (used in ParticleNet)

**Coordinate System:**
- Center jet at (η=0, φ=0) using pT-weighted centroid
- Use relative coordinates (Δη, Δφ) for translation invariance

**Node Features:**
- (Δη, Δφ, log(pT), log(E), pT/ΣpT)
- Optional: pdgid embedding

---

## 4. GNN Architecture Research

### 4.1 Candidate Architectures - Initial List

Based on literature and HEP applications, initial candidates:

1. **ParticleNet (EdgeConv/DGCNN)**
2. **Graph Attention Network (GAT)**
3. **Graph Isomorphism Network (GIN)**
4. **PointNet++ (hierarchical)**
5. **Message Passing Neural Network (MPNN)**
6. **Transformer-based (full attention)**
7. **LorentzNet (equivariant)**
8. **GraphSAGE**

### 4.2 Architecture Deep Dive

#### 4.2.1 ParticleNet / EdgeConv (DGCNN)

**Source:** Dynamic Graph CNN for Learning on Point Clouds (Wang et al., 2019)

**Core Operation:**
```
EdgeConv: h_i' = max_{j∈N(i)} MLP(h_i || h_j - h_i)
```
- Concatenates node features with edge features (difference)
- Max aggregation over neighbors
- Dynamic graph: k-NN recomputed in feature space each layer

**Architecture:**
```
Input → EdgeConv Block ×3 → Global Pooling → MLP → Output
Each EdgeConv: (k-NN, MLP, max-pool)
```

**Strengths:**
- State-of-the-art on jet tagging benchmarks
- Learns to construct meaningful graphs
- Captures local geometric structure

**Weaknesses:**
- k-NN computation at each layer is expensive
- No edge features beyond coordinate differences

#### 4.2.2 Graph Attention Network (GAT)

**Source:** Veličković et al., 2018

**Core Operation:**
```
α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
h_i' = σ(Σ_j α_ij W h_j)
```
- Attention weights determine neighbor importance
- Multi-head attention for stability

**Strengths:**
- Interpretable attention weights (which particles matter?)
- Adaptive aggregation
- Well-understood architecture

**Weaknesses:**
- Requires pre-defined graph structure
- Attention may focus too narrowly

#### 4.2.3 Graph Isomorphism Network (GIN)

**Source:** Xu et al., 2019 (How Powerful are GNNs?)

**Core Operation:**
```
h_i' = MLP((1 + ε) · h_i + Σ_{j∈N(i)} h_j)
```
- Sum aggregation (most expressive)
- Learnable ε parameter

**Strengths:**
- Maximally expressive (as powerful as WL test)
- Simple and efficient
- Strong theoretical foundation

**Weaknesses:**
- No edge features in basic form
- May overfit on small datasets

#### 4.2.4 Message Passing Neural Network (MPNN)

**Source:** Gilmer et al., 2017

**Core Operation:**
```
m_ij = M(h_i, h_j, e_ij)  # Message function
h_i' = U(h_i, Σ_j m_ij)   # Update function
```
- General framework encompassing many GNNs
- Explicitly uses edge features

**Strengths:**
- Flexible message and update functions
- Can incorporate edge features naturally
- Physics-informed design possible

**Weaknesses:**
- Many design choices to make
- Performance depends on specific instantiation

#### 4.2.5 TransformerConv / Graph Transformer

**Source:** Various, including Set Transformer, Perceiver

**Core Operation:**
```
Full self-attention over all nodes
h_i' = Σ_j softmax(Q_i · K_j / √d) · V_j
```

**Strengths:**
- Captures global context
- Very expressive
- No fixed graph structure needed

**Weaknesses:**
- O(n²) complexity
- May need careful regularization
- Loses locality inductive bias

#### 4.2.6 LorentzNet (Equivariant GNN)

**Source:** Gong et al., 2022

**Core Idea:**
- Respects Lorentz symmetry of particle physics
- Equivariant to boosts and rotations

**Strengths:**
- Physics-informed architecture
- Better generalization to different kinematic regions

**Weaknesses:**
- More complex implementation
- May be overkill for narrow pT window dataset

---

## 5. Iterative Architecture Selection

### 5.1 Selection Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Performance** | High | Expected accuracy on q/g classification |
| **Interpretability** | Medium | Can we understand what the model learns? |
| **Computational Cost** | Medium | Training and inference time |
| **Implementation Complexity** | Medium | Ease of implementation with PyG |
| **Novelty/Diversity** | Medium | Different approaches provide insights |

### 5.2 Round 1: Initial Filtering

**Eliminate based on clear criteria:**

| Architecture | Keep/Drop | Reason |
|--------------|-----------|--------|
| ParticleNet/EdgeConv | ✓ KEEP | State-of-the-art, proven on this exact task |
| GAT | ✓ KEEP | Interpretable, good performance |
| GIN | ✓ KEEP | Theoretically strongest expressiveness |
| PointNet++ | ✗ DROP | Hierarchical structure less suited for flat jets |
| MPNN | ✓ KEEP | Flexible, can use edge features |
| TransformerConv | ✓ KEEP | Global attention, alternative paradigm |
| LorentzNet | ✗ DROP | Complex, narrow pT window reduces need for equivariance |
| GraphSAGE | ✗ DROP | Designed for large graphs, less suited here |

**Remaining:** ParticleNet, GAT, GIN, MPNN, TransformerConv

### 5.3 Round 2: Detailed Comparison

#### Comparison Matrix

| Aspect | ParticleNet | GAT | GIN | MPNN | TransformerConv |
|--------|-------------|-----|-----|------|-----------------|
| HEP Track Record | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐ |
| Expressiveness | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Interpretability | ⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ |
| Efficiency | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Edge Features | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| PyG Support | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

#### Analysis:

**ParticleNet (EdgeConv):**
- ✅ Specifically designed for particle physics point clouds
- ✅ Published results on this exact dataset (AUC ~0.84)
- ✅ DynamicEdgeConv available in PyG
- ❌ Dynamic graph construction adds overhead

**GAT:**
- ✅ Attention weights provide interpretability
- ✅ Efficient implementation in PyG (GATConv, GATv2Conv)
- ✅ Good baseline performance
- ❌ Requires static graph (k-NN constructed once)

**GIN:**
- ✅ Theoretically most expressive
- ✅ Very simple and efficient
- ❌ No native edge feature support
- ❌ May be too simple for this task

**MPNN:**
- ✅ Explicit edge features (physically meaningful)
- ✅ Flexible architecture
- ❌ Generic - need to specify message/update functions
- Essentially a framework, will instantiate as EdgeConv or similar

**TransformerConv:**
- ✅ Global attention captures long-range dependencies
- ✅ Alternative paradigm to local message passing
- ❌ O(n²) scaling problematic for high multiplicity jets
- ❌ Less physics-motivated

### 5.4 Round 3: Final Selection

**Selection Rationale:**

For this task, I want to select two architectures that:
1. Provide complementary approaches (local vs global, static vs dynamic)
2. Have proven track records or strong theoretical motivation
3. Allow for meaningful comparison and insights

**Architecture 1: ParticleNet (Dynamic EdgeConv)**
- **Rationale:** State-of-the-art for this exact task. Dynamic graph learning is a unique strength. Represents the best of local message passing with adaptive structure.
- **Expected Performance:** AUC ~0.83-0.85

**Architecture 2: Graph Attention Network (GAT) with Edge Features (GATv2)**
- **Rationale:** Provides interpretability through attention weights. Static k-NN graph provides efficiency. Different paradigm from EdgeConv (attention vs max-pool). GATv2 fixes expressiveness issues of original GAT.
- **Expected Performance:** AUC ~0.81-0.83

**Why not GIN?**
- While theoretically powerful, it lacks edge feature support and attention mechanism
- Less interpretable for physics analysis
- GAT provides similar expressiveness with added interpretability

**Why not TransformerConv?**
- O(n²) scaling is problematic for jets with 50+ particles
- Full attention may introduce spurious long-range correlations
- Less physics-motivated for local jet structure

### 5.5 Architecture Design Decisions

#### Architecture 1: ParticleNet (EdgeConv-based)

```
Input Features:
  - Coordinates: (Δη, Δφ) for k-NN
  - Node features: (Δη, Δφ, log(pT), log(E), pT/ΣpT)

Architecture:
  EdgeConv Block 1: k=16, MLP(10 → 64 → 64 → 64)
  EdgeConv Block 2: k=16, MLP(64 → 128 → 128 → 128)  
  EdgeConv Block 3: k=16, MLP(128 → 256 → 256 → 256)
  
  Global Pooling: mean + max concatenation
  
  Classifier:
    Linear(512 → 256) → ReLU → Dropout(0.3)
    Linear(256 → 128) → ReLU → Dropout(0.3)  
    Linear(128 → 2)
```

#### Architecture 2: GAT with Static k-NN Graph

```
Input Features:
  - Node features: (Δη, Δφ, log(pT), log(E), pT/ΣpT)
  - Edge features: (ΔR_ij, Δη_ij, Δφ_ij)

Graph Construction:
  - k-NN with k=16 in (Δη, Δφ) space (static, computed once)

Architecture:
  GATv2Conv Block 1: (5 → 64), heads=4, concat=True → 256
  GATv2Conv Block 2: (256 → 64), heads=4, concat=True → 256
  GATv2Conv Block 3: (256 → 64), heads=4, concat=False → 64
  
  Global Pooling: mean + max concatenation
  
  Classifier:
    Linear(128 → 128) → ReLU → Dropout(0.3)
    Linear(128 → 64) → ReLU → Dropout(0.3)
    Linear(64 → 2)
```

---

## 6. Final Architecture Decisions

### 6.1 Summary Table

| Aspect | ParticleNet (EdgeConv) | GAT (GATv2Conv) |
|--------|------------------------|-----------------|
| **Graph Type** | Dynamic k-NN (recomputed each layer) | Static k-NN (computed once) |
| **Message Passing** | Edge features + MLP + max-pool | Attention-weighted sum |
| **Key Strength** | Learns optimal graph structure | Interpretable attention weights |
| **Parameters** | ~500K | ~300K |
| **Training Time** | Slower (k-NN overhead) | Faster |

### 6.2 Graph Construction (Shared)

Both architectures will use:
- **k-NN graph with k=16** in (Δη, Δφ) space
- **Node features:** (Δη, Δφ, log(pT), log(E), pT_frac)
- **Jet centering:** pT-weighted centroid at origin

**ParticleNet** will recompute k-NN in learned feature space after each layer.
**GAT** will use a static graph constructed once from input coordinates.

### 6.3 Hyperparameters

| Hyperparameter | ParticleNet | GAT |
|----------------|-------------|-----|
| k (neighbors) | 16 | 16 |
| Hidden dim | 64 → 128 → 256 | 64 × 4 heads |
| Layers | 3 EdgeConv | 3 GATv2Conv |
| Dropout | 0.3 | 0.3 |
| Learning rate | 1e-3 | 1e-3 |
| Batch size | 128 | 128 |
| Optimizer | Adam | Adam |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |

---

## 7. Implementation Plan

### 7.1 Code Structure

```
Task 2/
├── PLANNING.md                 # This document
├── DOCUMENTATION.md            # Final refined documentation
├── README.md                   # Task description
├── requirements.txt            # Dependencies
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # JetDataset class, data loading
│   │   └── preprocessing.py    # Feature engineering, graph construction
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── particle_net.py     # ParticleNet (EdgeConv) model
│   │   ├── gat_classifier.py   # GAT model
│   │   └── layers.py           # Custom layers if needed
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop, validation
│   │   └── metrics.py          # AUC, accuracy, etc.
│   │
│   └── utils/
│       ├── __init__.py
│       └── visualization.py    # Plotting, attention visualization
│
├── notebooks/
│   └── exploration.ipynb       # Data exploration (optional)
│
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
└── results/                    # Saved models, plots, metrics
```

### 7.2 Implementation Steps

1. **Data Pipeline**
   - Load data using EnergyFlow package
   - Implement preprocessing (centering, feature engineering)
   - Create PyG Dataset and DataLoader
   - Handle variable-size jets (padding removal)

2. **Model Implementation**
   - Implement ParticleNet using DynamicEdgeConv from PyG
   - Implement GAT classifier using GATv2Conv
   - Add global pooling and classification heads

3. **Training Pipeline**
   - Implement training loop with validation
   - Add metrics: AUC, accuracy, loss curves
   - Implement early stopping and checkpointing

4. **Evaluation**
   - ROC curves comparison
   - Confusion matrices
   - Attention visualization for GAT
   - Performance analysis

### 7.3 Dependencies

```
torch>=2.0
torch-geometric>=2.3
energyflow
numpy
scikit-learn
matplotlib
tqdm
```

---

## Research Sources

1. **ParticleNet:** H. Qu, L. Gouskos, "ParticleNet: Jet Tagging via Particle Clouds", arXiv:1902.08570
2. **DGCNN:** Y. Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", arXiv:1801.07829
3. **GAT:** P. Veličković et al., "Graph Attention Networks", arXiv:1710.10903
4. **GATv2:** S. Brody et al., "How Attentive are Graph Attention Networks?", arXiv:2105.14491
5. **GIN:** K. Xu et al., "How Powerful are Graph Neural Networks?", arXiv:1810.00826
6. **EnergyFlow:** P. Komiske et al., "Energy Flow Networks", arXiv:1810.05165
7. **Jet Tagging Review:** A. Larkoski et al., "Jet Substructure at the LHC", arXiv:1709.04464
8. **PyTorch Geometric Documentation:** https://pytorch-geometric.readthedocs.io/

---

*Document created: Planning phase for Task II GNN implementation*
*Next step: Implement the modular code pipeline*
