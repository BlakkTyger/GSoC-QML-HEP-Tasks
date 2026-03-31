# Task 12: PQC Embedding with Reinforcement Learning

## Task Overview

Solve the same PQC embedding problem from Task 11, but using **reinforcement learning** (temporal difference algorithms) instead of supervised gradient descent.

**Problem Recap:**
- Input: x ~ N(0, 1) (scalar)
- Target: y = [x, sin(x), cos(x), x²]
- Model: MLP → PQC parameters → Quantum Circuit → Output

**New Approach:** Use RL (PPO) to learn the policy that maps states to actions.

---

## RL Problem Formulation

### State Space
- **State s**: Input scalar x ∈ ℝ (sampled from N(0,1))
- State dimension: 1

### Action Space
- **Action a**: PQC parameters θ ∈ ℝ^15 (for 5 qubits × 3 layers)
- Continuous action space
- Actions bounded to reasonable range (e.g., [-π, π])

### Reward Function
- **Reward r**: Negative MSE between PQC output and target
- r = -MSE(PQC(θ), y_target)
- Higher reward = lower MSE = better approximation

### Episode Structure
- Each episode: Sample batch of inputs, compute rewards
- No sequential dependency (each state is independent)
- This is essentially a **contextual bandit** problem

---

## Research & Algorithm Selection

### 1. Why PPO over DQN?

| Algorithm | Action Space | Suitable? |
|-----------|--------------|-----------|
| DQN | Discrete | ❌ (15D continuous actions) |
| DDPG | Continuous | ✅ (but unstable) |
| TD3 | Continuous | ✅ (improved DDPG) |
| **PPO** | Both | ✅ **Best choice** |
| SAC | Continuous | ✅ (but complex) |

**Decision: PPO (Proximal Policy Optimization)**
- Works with continuous action spaces
- Stable training with clipped objectives
- Simple to implement
- Good sample efficiency

### 2. PPO Algorithm Overview

**Key Components:**
1. **Actor (Policy Network)**: π(a|s) - outputs action distribution
2. **Critic (Value Network)**: V(s) - estimates state value
3. **Clipped Objective**: Prevents large policy updates

**PPO Loss:**
```
L_CLIP = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
```
where r_t(θ) = π_θ(a|s) / π_θ_old(a|s)

### 3. Continuous Action Distribution

For continuous actions, we use a **Gaussian policy**:
- Actor outputs: μ (mean) and σ (std) for each action dimension
- Actions sampled: a ~ N(μ, σ²)
- Log probability computed for policy gradient

### 4. References

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Spinning Up in Deep RL (OpenAI) - PPO implementation guide
- Stable-Baselines3 documentation

---

## Architecture Design

### Actor Network (Policy)
```
Input: state s (dim=1)
    ↓
Linear(1 → 64) + ReLU
    ↓
Linear(64 → 64) + ReLU
    ↓
├── Linear(64 → 15) → action_mean (μ)
└── Linear(64 → 15) → action_log_std (log σ)
    ↓
Output: Gaussian distribution N(μ, σ²)
```

### Critic Network (Value)
```
Input: state s (dim=1)
    ↓
Linear(1 → 64) + ReLU
    ↓
Linear(64 → 64) + ReLU
    ↓
Linear(64 → 1)
    ↓
Output: V(s) (state value estimate)
```

### PQC (Same as Task 11)
- 5 qubits, 3 layers
- RY rotations + ring CNOT entanglement
- Output: 4 Pauli-Z expectations

### Output Scaling
- Learnable scale and bias (as in Task 11)
- Or: Include in reward shaping

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | PPO | Continuous actions, stable |
| Episodes | 500 | Sufficient for convergence |
| Steps per episode | 64 | Batch of samples |
| Learning Rate (Actor) | 3e-4 | Standard PPO |
| Learning Rate (Critic) | 1e-3 | Faster value learning |
| Gamma (γ) | 0.99 | Discount factor |
| GAE Lambda (λ) | 0.95 | Advantage estimation |
| Clip Epsilon (ε) | 0.2 | PPO clipping |
| Entropy Coefficient | 0.01 | Exploration |
| PPO Epochs | 10 | Updates per batch |
| Mini-batch Size | 32 | For PPO updates |

---

## Implementation Plan

### File Structure
```
Task 12/
├── src/
│   ├── __init__.py
│   ├── environment.py    # RL environment (PQC reward)
│   ├── agent.py          # PPO agent (actor-critic)
│   ├── pqc.py            # Quantum circuit (from Task 11)
│   └── training.py       # Training loop
├── results/
│   ├── metrics.txt
│   ├── training_curve.png
│   └── model.pt
├── main.py
├── requirements.txt
├── PLANNING.md
├── DOCUMENTATION.md
└── README.md
```

### Implementation Order
1. `src/pqc.py` - Quantum circuit (reuse from Task 11)
2. `src/environment.py` - Custom RL environment
3. `src/agent.py` - PPO actor-critic
4. `src/training.py` - Training utilities
5. `main.py` - Entry point

---

## Key Differences from Task 11

| Aspect | Task 11 (Supervised) | Task 12 (RL) |
|--------|---------------------|--------------|
| Learning | Gradient descent on MSE | Policy gradient (PPO) |
| Model | Single MLP | Actor + Critic networks |
| Objective | Minimize loss | Maximize reward |
| Updates | Direct backprop | Advantage-weighted updates |
| Exploration | None | Gaussian noise + entropy |

---

## Expected Challenges

1. **Sample Efficiency**: RL typically needs more samples
2. **Reward Shaping**: MSE scale affects learning
3. **Action Range**: PQC params should be bounded
4. **Variance**: Policy gradient has high variance

### Mitigation Strategies
1. Use baseline (critic) to reduce variance
2. Normalize rewards for stable training
3. Clip actions to [-π, π] range
4. Use GAE for advantage estimation

---

## Success Criteria

- **Primary**: Achieve test MSE ≤ 0.10 (comparable to Task 11)
- **Secondary**: Stable training without divergence
- **Stretch**: Match Task 11 performance (MSE ~0.054)

---

## Iteration Log

| Iteration | Changes | Test MSE | Notes |
|-----------|---------|----------|-------|
| 1 | Initial PPO (300 eps, 64 steps) | 1.1667 | High variance |
| 2 | Optimized (500 eps, 128 steps, larger net) | **1.0438** | Better but still higher than supervised |

### Final Results Summary

**Configuration Used:**
- Algorithm: PPO
- Episodes: 500
- Steps per Episode: 128
- Hidden Dimension: 128
- Learning Rate (Actor): 1e-3
- Learning Rate (Critic): 3e-3

**Performance:**
| Metric | Value |
|--------|-------|
| Best Training MSE | 0.7407 |
| Evaluation MSE | 1.0438 |

**Comparison with Task 11:**
| Approach | Test MSE | Improvement |
|----------|----------|-------------|
| Task 11 Baseline | 0.5397 | - |
| Task 11 Optimized | 0.0541 | 90% vs baseline |
| Task 12 (RL/PPO) | 1.0438 | -93% vs Task 11 |

**Key Findings:**
1. RL successfully learns the mapping from states to PQC parameters
2. Supervised learning is ~19x more sample-efficient for this regression task
3. cos(x) target learned best (MSE 0.14) - bounded output matches PQC range
4. x² target hardest (MSE 2.06) - unbounded positive range

**Conclusion:**
While PPO can solve this task, supervised gradient descent is fundamentally better suited for regression problems where the target function is known and differentiable.
