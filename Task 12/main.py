"""Main script for Task 12: PQC Embedding with Reinforcement Learning (PPO)."""

import os
import torch
import argparse
import numpy as np

from src.environment import PQCEnvironment
from src.agent import PPOAgent
from src.training import train_ppo, evaluate_agent, plot_training_history
from src.pqc import N_QUBITS, N_LAYERS, N_PQC_PARAMS

torch.set_default_dtype(torch.float64)


def main(args):
    """Main training and evaluation pipeline."""
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Task 12: PQC Embedding with Reinforcement Learning (PPO)")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  - Qubits: {N_QUBITS}")
    print(f"  - PQC Layers: {N_LAYERS}")
    print(f"  - PQC Parameters: {N_PQC_PARAMS}")
    print(f"  - Episodes: {args.episodes}")
    print(f"  - Steps per Episode: {args.steps_per_episode}")
    print(f"  - Learning Rate (Actor): {args.lr_actor}")
    print(f"  - Learning Rate (Critic): {args.lr_critic}")
    
    print("\n[1/4] Initializing environment and agent...")
    
    env = PQCEnvironment(
        action_scale=np.pi,
        reward_scale=1.0,
        seed=args.seed
    )
    
    agent = PPOAgent(
        state_dim=1,
        action_dim=N_PQC_PARAMS,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        action_scale=np.pi
    )
    
    total_params = sum(p.numel() for p in agent.network.parameters())
    print(f"  - Agent parameters: {total_params}")
    
    print("\n[2/4] Training PPO agent...")
    print("-" * 60)
    
    history = train_ppo(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        verbose=True
    )
    
    print("-" * 60)
    print(f"\nBest Training MSE: {history['best_mse']:.4f}")
    
    print("\n[3/4] Evaluating agent...")
    
    eval_results = evaluate_agent(agent, n_samples=500, seed=123)
    
    print(f"\nEvaluation Results:")
    print(f"  - Total MSE: {eval_results['total_mse']:.4f}")
    print(f"  - Per-target MSE:")
    for name, mse in eval_results['per_target_mse'].items():
        print(f"      {name}: {mse:.4f}")
    
    print("\n[4/4] Saving results...")
    
    plot_path = os.path.join(args.results_dir, "training_curve.png")
    plot_training_history(history, save_path=plot_path)
    
    model_path = os.path.join(args.results_dir, "agent.pt")
    agent.save(model_path)
    print(f"Agent saved to {model_path}")
    
    metrics_path = os.path.join(args.results_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Task 12: PQC Embedding with RL (PPO) Results\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Algorithm: PPO\n")
        f.write(f"  Qubits: {N_QUBITS}\n")
        f.write(f"  PQC Layers: {N_LAYERS}\n")
        f.write(f"  Hidden Dim: {args.hidden_dim}\n")
        f.write(f"  Episodes: {args.episodes}\n")
        f.write(f"  Steps per Episode: {args.steps_per_episode}\n")
        f.write(f"  LR Actor: {args.lr_actor}\n")
        f.write(f"  LR Critic: {args.lr_critic}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Best Training MSE: {history['best_mse']:.4f}\n")
        f.write(f"  Final Eval MSE: {eval_results['total_mse']:.4f}\n\n")
        f.write(f"Per-Target MSE:\n")
        for name, mse in eval_results['per_target_mse'].items():
            f.write(f"  {name}: {mse:.4f}\n")
        f.write(f"\nComparison:\n")
        f.write(f"  Task 11 Baseline (Supervised): 0.5397\n")
        f.write(f"  Task 11 Best (Supervised): 0.0541\n")
        f.write(f"  Task 12 (RL/PPO): {eval_results['total_mse']:.4f}\n")
    print(f"Metrics saved to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return history, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 12: PQC Embedding with PPO")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes")
    parser.add_argument("--steps_per_episode", type=int, default=64, help="Steps per episode")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    
    args = parser.parse_args()
    main(args)
