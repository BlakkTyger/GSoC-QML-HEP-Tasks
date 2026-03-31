"""PPO Agent for PQC parameter learning."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, List
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor: Outputs mean and log_std for Gaussian policy
    Critic: Outputs state value estimate
    """
    
    def __init__(
        self,
        state_dim: int = 1,
        action_dim: int = 15,
        hidden_dim: int = 64,
        action_scale: float = np.pi
    ):
        super(ActorCritic, self).__init__()
        
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.output_scale = nn.Parameter(torch.ones(4))
        self.output_bias = nn.Parameter(torch.zeros(4))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action mean and value.
        
        Args:
            state: State tensor of shape (batch, state_dim)
        
        Returns:
            Tuple of (action_mean, value)
        """
        actor_features = self.actor_backbone(state)
        action_mean = self.actor_mean(actor_features)
        action_mean = torch.tanh(action_mean) * self.action_scale
        
        value = self.critic(state)
        return action_mean, value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_mean, value = self.forward(state)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action.shape[0], device=action.device)
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        action = torch.clamp(action, -self.action_scale, self.action_scale)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions.
        
        Args:
            states: Batch of states
            actions: Batch of actions
        
        Returns:
            Tuple of (log_prob, entropy, value)
        """
        action_mean, value = self.forward(states)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value.squeeze(-1)


class PPOAgent:
    """
    PPO Agent for learning PQC parameters.
    """
    
    def __init__(
        self,
        state_dim: int = 1,
        action_dim: int = 15,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        action_scale: float = np.pi
    ):
        """
        Initialize PPO agent.
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.network = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_scale=action_scale
        ).double()
        
        self.optimizer = optim.Adam([
            {'params': self.network.actor_backbone.parameters(), 'lr': lr_actor},
            {'params': self.network.actor_mean.parameters(), 'lr': lr_actor},
            {'params': [self.network.actor_log_std], 'lr': lr_actor},
            {'params': self.network.critic.parameters(), 'lr': lr_critic},
            {'params': [self.network.output_scale, self.network.output_bias], 'lr': lr_critic}
        ])
    
    def select_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action given state."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            action, log_prob, value = self.network.get_action(state, deterministic)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float64)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float64)
        
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        ppo_epochs: int = 10,
        mini_batch_size: int = 32
    ) -> dict:
        """
        Perform PPO update.
        
        Returns:
            Dictionary with loss statistics
        """
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        dataset_size = states.shape[0]
        
        for _ in range(ppo_epochs):
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, mini_batch_size):
                end = min(start + mini_batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                log_probs, entropy, values = self.network.evaluate_actions(
                    batch_states, batch_actions
                )
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(values, batch_returns)
                
                entropy_loss = -entropy.mean()
                
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates
        }
    
    def get_output_scaling(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current output scaling parameters."""
        return self.network.output_scale, self.network.output_bias
    
    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
