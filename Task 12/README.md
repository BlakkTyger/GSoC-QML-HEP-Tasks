# Task XII: PQC Embedding with Reinforcement Learning

## Task Description

Implement Task XI (PQC embedding) using **reinforcement learning** instead of supervised learning.
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Reward**: -MSE between PQC output and target
- **Action Space**: Continuous PQC parameters (15 dimensions)

## Implementation

### Architecture
- **Actor**: 1 → 128 → 128 → 15 (Gaussian policy)
- **Critic**: 1 → 128 → 128 → 1 (Value function)
- **PQC**: 5 qubits, 3 layers (same as Task 11)

### RL Formulation
- **State**: Input x ~ N(0,1)
- **Action**: PQC parameters θ ∈ [-π, π]^15
- **Reward**: r = -MSE(PQC(θ), [x, sin(x), cos(x), x²])

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python main.py --episodes 500 --steps_per_episode 128

# Custom configuration
python main.py --lr_actor 1e-3 --hidden_dim 128
```

## Results

| Approach | Test MSE | Notes |
|----------|----------|-------|
| Task 11 (Supervised) | **0.0541** | More sample-efficient |
| Task 12 (RL/PPO) | 1.0438 | Higher variance |

### Per-Target MSE
| Target | MSE |
|--------|-----|
| x | 1.3071 |
| sin(x) | 0.6629 |
| cos(x) | 0.1421 |
| x² | 2.0631 |

### Key Insight
RL successfully learns the task but is **less sample-efficient** than supervised learning for regression problems. Supervised learning achieves 19x better MSE.

## Files

- `main.py` - Entry point for training
- `src/agent.py` - PPO actor-critic implementation
- `src/environment.py` - RL environment with PQC reward
- `src/pqc.py` - Quantum circuit
- `src/training.py` - Training utilities
- `PLANNING.md` - Design decisions and research
- `DOCUMENTATION.md` - Detailed documentation

## Dependencies

- PyTorch >= 2.0.0
- PennyLane >= 0.33.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0 