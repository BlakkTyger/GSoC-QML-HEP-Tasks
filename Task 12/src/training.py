"""Training and evaluation utilities for PPO agent."""

import torch
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from .environment import PQCEnvironment
from .agent import PPOAgent
from .pqc import compute_target, run_pqc


def collect_rollout(
    env: PQCEnvironment,
    agent: PPOAgent,
    n_steps: int
) -> dict:
    """
    Collect a rollout of experiences.
    
    Args:
        env: PQC environment
        agent: PPO agent
        n_steps: Number of steps to collect
    
    Returns:
        Dictionary containing rollout data
    """
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    mse_values = []
    
    state = env.reset()
    scale, bias = agent.get_output_scaling()
    env.update_output_scaling(scale, bias)
    
    for _ in range(n_steps):
        action, log_prob, value = agent.select_action(state)
        
        next_state, reward, done, info = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value.item())
        dones.append(done)
        mse_values.append(info["mse"])
        
        state = next_state
    
    _, _, next_value = agent.select_action(state)
    
    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "rewards": rewards,
        "log_probs": torch.stack(log_probs),
        "values": values,
        "dones": dones,
        "next_value": next_value.item(),
        "mse_values": mse_values
    }


def train_ppo(
    env: PQCEnvironment,
    agent: PPOAgent,
    n_episodes: int = 500,
    steps_per_episode: int = 64,
    ppo_epochs: int = 10,
    mini_batch_size: int = 32,
    verbose: bool = True
) -> dict:
    """
    Train PPO agent.
    
    Args:
        env: PQC environment
        agent: PPO agent
        n_episodes: Number of training episodes
        steps_per_episode: Steps to collect per episode
        ppo_epochs: PPO update epochs per rollout
        mini_batch_size: Mini-batch size for updates
        verbose: Whether to print progress
    
    Returns:
        Training history dictionary
    """
    history = {
        "episode_rewards": [],
        "episode_mse": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": []
    }
    
    best_mse = float("inf")
    best_state_dict = None
    
    for episode in range(n_episodes):
        rollout = collect_rollout(env, agent, steps_per_episode)
        
        advantages, returns = agent.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            rollout["next_value"]
        )
        
        update_stats = agent.update(
            rollout["states"],
            rollout["actions"],
            rollout["log_probs"],
            advantages,
            returns,
            ppo_epochs=ppo_epochs,
            mini_batch_size=mini_batch_size
        )
        
        avg_reward = np.mean(rollout["rewards"])
        avg_mse = np.mean(rollout["mse_values"])
        
        history["episode_rewards"].append(avg_reward)
        history["episode_mse"].append(avg_mse)
        history["policy_loss"].append(update_stats["policy_loss"])
        history["value_loss"].append(update_stats["value_loss"])
        history["entropy"].append(update_stats["entropy"])
        
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_state_dict = {k: v.clone() for k, v in agent.network.state_dict().items()}
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:4d}/{n_episodes} | "
                  f"Reward: {avg_reward:8.4f} | "
                  f"MSE: {avg_mse:.4f} | "
                  f"Policy Loss: {update_stats['policy_loss']:.4f}")
    
    if best_state_dict is not None:
        agent.network.load_state_dict(best_state_dict)
    
    history["best_mse"] = best_mse
    
    return history


def evaluate_agent(
    agent: PPOAgent,
    n_samples: int = 500,
    seed: int = 123
) -> dict:
    """
    Evaluate trained agent.
    
    Args:
        agent: Trained PPO agent
        n_samples: Number of test samples
        seed: Random seed for reproducibility
    
    Returns:
        Evaluation results dictionary
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    states = torch.randn(n_samples, 1, dtype=torch.float64)
    
    scale, bias = agent.get_output_scaling()
    
    predictions = []
    targets = []
    mse_values = []
    
    agent.network.eval()
    with torch.no_grad():
        for i in range(n_samples):
            state = states[i:i+1]
            target = compute_target(state)
            
            action, _, _ = agent.network.get_action(state, deterministic=True)
            pqc_output = run_pqc(action.squeeze(0))
            scaled_output = pqc_output * scale + bias
            
            mse = torch.mean((scaled_output - target) ** 2).item()
            
            predictions.append(scaled_output)
            targets.append(target)
            mse_values.append(mse)
    
    predictions = torch.stack(predictions)
    targets = torch.stack(targets)
    
    total_mse = np.mean(mse_values)
    
    target_names = ["x", "sin(x)", "cos(x)", "x²"]
    per_target_mse = {}
    for i, name in enumerate(target_names):
        per_target_mse[name] = torch.mean((predictions[:, i] - targets[:, i]) ** 2).item()
    
    return {
        "total_mse": total_mse,
        "per_target_mse": per_target_mse,
        "predictions": predictions,
        "targets": targets
    }


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(history["episode_rewards"], color="blue", alpha=0.7)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Average Reward")
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history["episode_mse"], color="red", alpha=0.7)
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_title("Episode MSE")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history["policy_loss"], label="Policy", color="green", alpha=0.7)
    axes[1, 0].plot(history["value_loss"], label="Value", color="orange", alpha=0.7)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Training Losses")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history["entropy"], color="purple", alpha=0.7)
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].set_title("Policy Entropy")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    plt.close()
