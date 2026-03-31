"""RL Environment for PQC parameter learning."""

import torch
import numpy as np
from typing import Tuple

from .pqc import run_pqc, compute_target, N_PQC_PARAMS, OUTPUT_DIM


class PQCEnvironment:
    """
    Custom RL environment for PQC embedding.
    
    State: Input x ~ N(0, 1)
    Action: PQC parameters (continuous, 15-dimensional)
    Reward: -MSE between PQC output and target
    """
    
    def __init__(
        self,
        action_scale: float = np.pi,
        reward_scale: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize the environment.
        
        Args:
            action_scale: Scale for action bounds [-scale, scale]
            reward_scale: Scale factor for rewards
            seed: Random seed
        """
        self.action_scale = action_scale
        self.reward_scale = reward_scale
        self.state_dim = 1
        self.action_dim = N_PQC_PARAMS
        self.output_dim = OUTPUT_DIM
        
        self.output_scale = torch.ones(OUTPUT_DIM, dtype=torch.float64)
        self.output_bias = torch.zeros(OUTPUT_DIM, dtype=torch.float64)
        
        self.rng = np.random.RandomState(seed)
        self.current_state = None
        self.current_target = None
    
    def reset(self) -> torch.Tensor:
        """
        Reset environment with new random state.
        
        Returns:
            New state tensor of shape (1,)
        """
        x = self.rng.randn()
        self.current_state = torch.tensor([x], dtype=torch.float64)
        self.current_target = compute_target(self.current_state)
        return self.current_state
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Execute action and compute reward.
        
        Args:
            action: PQC parameters of shape (15,)
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        action = torch.clamp(action, -self.action_scale, self.action_scale)
        
        pqc_output = run_pqc(action)
        scaled_output = pqc_output * self.output_scale + self.output_bias
        
        mse = torch.mean((scaled_output - self.current_target) ** 2).item()
        reward = -mse * self.reward_scale
        
        next_state = self.reset()
        done = False
        
        info = {
            "mse": mse,
            "pqc_output": pqc_output.detach(),
            "target": self.current_target.detach()
        }
        
        return next_state, reward, done, info
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of states and targets.
        
        Args:
            batch_size: Number of samples
        
        Returns:
            Tuple of (states, targets) tensors
        """
        states = torch.randn(batch_size, 1, dtype=torch.float64)
        targets = torch.stack([compute_target(s) for s in states])
        return states, targets
    
    def compute_batch_reward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards for a batch of state-action pairs.
        
        Args:
            states: Batch of states (batch_size, 1)
            actions: Batch of actions (batch_size, 15)
        
        Returns:
            Tuple of (rewards, mse_values) tensors
        """
        batch_size = states.shape[0]
        rewards = []
        mse_values = []
        
        for i in range(batch_size):
            target = compute_target(states[i])
            action = torch.clamp(actions[i], -self.action_scale, self.action_scale)
            pqc_output = run_pqc(action)
            scaled_output = pqc_output * self.output_scale + self.output_bias
            
            mse = torch.mean((scaled_output - target) ** 2)
            reward = -mse * self.reward_scale
            
            rewards.append(reward)
            mse_values.append(mse)
        
        return torch.stack(rewards), torch.stack(mse_values)
    
    def update_output_scaling(self, scale: torch.Tensor, bias: torch.Tensor):
        """Update output scaling parameters."""
        self.output_scale = scale.detach()
        self.output_bias = bias.detach()
