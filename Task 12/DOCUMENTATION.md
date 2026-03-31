# Task 12: PQC Embedding with Reinforcement Learning

## Overview

This task implements the same PQC embedding problem from Task 11, but using **Reinforcement Learning (PPO)** instead of supervised gradient descent. The goal is to learn a policy that maps input states to PQC parameters, using MSE as the reward signal.

---

## Problem Formulation

### From Supervised Learning to RL

| Aspect | Task 11 (Supervised) | Task 12 (RL) |
|--------|---------------------|--------------|
| **Input** | x ~ N(0,1) | State s = x |
| **Output** | PQC parameters | Action a = θ |
| **Objective** | Minimize MSE loss | Maximize reward |
| **Learning** | Direct backpropagation | Policy gradient |
| **Model** | Single MLP | Actor + Critic |

### RL Components

- **State**: Input scalar x ∈ ℝ (from N(0,1))
- **Action**: PQC parameters θ ∈ ℝ^15 (continuous, bounded to [-π, π])
- **Reward**: r = -MSE(PQC(θ), target)
- **Target**: y = [x, sin(x), cos(x), x²]

---

## Architecture

### PPO Agent

```
┌─────────────────────────────────────────────────────────────┐
│                    PPO ACTOR-CRITIC                         │
├─────────────────────────────────────────────────────────────┤
│  State s (input x)                                          │
│         │                                                   │
│         ├──────────────┬──────────────┐                     │
│         ▼              ▼              │                     │
│  ┌────────────┐  ┌────────────┐       │                     │
│  │   ACTOR    │  │   CRITIC   │       │                     │
│  │ 1→64→64→15 │  │ 1→64→64→1  │       │                     │
│  └────────────┘  └────────────┘       │                     │
│         │              │              │                     │
│         ▼              ▼              │                     │
│   Action mean μ    Value V(s)         │                     │
│   + log_std σ                         │                     │
│         │                             │                     │
│         ▼                             │                     │
│  Sample a ~ N(μ, σ²)                  │                     │
│         │                             │                     │
│         ▼                             │                     │
│     PQC(a) → output                   │                     │
│         │                             │                     │
│         ▼                             │                     │
│  Reward = -MSE(output, target)        │                     │
└─────────────────────────────────────────────────────────────┘
```

### PQC (Same as Task 11)
- 5 qubits, 3 layers
- RY rotations + ring CNOT entanglement
- 4 Pauli-Z expectation outputs

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Episodes | 500 |
| Steps per Episode | 128 |
| Hidden Dimension | 128 |
| Actor Learning Rate | 1e-3 |
| Critic Learning Rate | 3e-3 |
| Discount (γ) | 0.99 |
| GAE Lambda (λ) | 0.95 |
| Clip Epsilon (ε) | 0.2 |
| Entropy Coefficient | 0.005 |
| PPO Epochs | 10 |
| Mini-batch Size | 32 |

---

## Results

### Performance Summary

| Metric | Value |
|--------|-------|
| Best Training MSE | 0.7407 |
| Evaluation MSE | 1.0438 |

### Per-Target MSE

| Target | MSE |
|--------|-----|
| x | 1.3071 |
| sin(x) | 0.6629 |
| cos(x) | 0.1421 |
| x² | 2.0631 |

### Comparison with Task 11

| Approach | Test MSE | Notes |
|----------|----------|-------|
| Task 11 Baseline (Supervised) | 0.5397 | Original notebook |
| Task 11 Optimized (Supervised) | 0.0541 | Our improved version |
| **Task 12 (RL/PPO)** | **1.0438** | This implementation |

---

## Analysis

### Why RL Underperforms Supervised Learning

1. **Sample Efficiency**: RL requires many more samples to converge
2. **High Variance**: Policy gradient methods have inherently high variance
3. **Exploration vs Exploitation**: RL must balance exploration, adding noise
4. **Credit Assignment**: Difficult to attribute reward to specific parameter choices
5. **Problem Structure**: This is essentially a regression task - supervised learning is more natural

### What RL Learned Well

- **cos(x)**: MSE 0.1421 - bounded output matches PQC range well
- **sin(x)**: MSE 0.6629 - reasonably approximated

### What RL Struggled With

- **x**: MSE 1.3071 - unbounded, requires scaling
- **x²**: MSE 2.0631 - unbounded positive, hardest target

---

## File Structure

```
Task 12/
├── src/
│   ├── __init__.py       # Package exports
│   ├── pqc.py            # Quantum circuit
│   ├── environment.py    # RL environment
│   ├── agent.py          # PPO actor-critic
│   └── training.py       # Training utilities
├── results/
│   ├── agent.pt          # Saved agent
│   ├── metrics.txt       # Evaluation metrics
│   └── training_curve.png
├── main.py               # Entry point
├── requirements.txt
├── PLANNING.md
├── DOCUMENTATION.md
└── README.md
```

---

## Usage

### Training
```bash
python main.py --episodes 500 --steps_per_episode 128
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 300 | Number of training episodes |
| `--steps_per_episode` | 64 | Steps per episode |
| `--hidden_dim` | 64 | Hidden layer dimension |
| `--lr_actor` | 3e-4 | Actor learning rate |
| `--lr_critic` | 1e-3 | Critic learning rate |
| `--entropy_coef` | 0.01 | Entropy bonus coefficient |

---

## Key Insights

1. **RL Can Solve This Task**: PPO successfully learns to map states to PQC parameters
2. **Supervised Learning is Better**: For regression tasks, direct gradient descent is more efficient
3. **RL Adds Value When**: 
   - Environment dynamics are unknown
   - Sequential decisions matter
   - Reward signal is sparse or delayed
4. **This Task**: Is essentially a contextual bandit - RL overhead is unnecessary

---

## Dependencies

- PyTorch >= 2.0.0
- PennyLane >= 0.33.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0

---

## References

1. Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
2. OpenAI Spinning Up - PPO Implementation Guide
3. Stable-Baselines3 Documentation
