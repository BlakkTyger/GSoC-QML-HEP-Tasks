"""Main script for Task 11: PQC Embedding with MLP Parameter Estimation."""

import os
import torch
import argparse

from src.dataset import generate_data, create_dataloaders
from src.model import HybridModel, N_QUBITS, N_LAYERS
from src.training import train_model, plot_training_history, detailed_evaluation

torch.set_default_dtype(torch.float64)


def main(args):
    """Main training and evaluation pipeline."""
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("=" * 60)
    print("Task 11: PQC Embedding with MLP Parameter Estimation")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  - Qubits: {N_QUBITS}")
    print(f"  - PQC Layers: {N_LAYERS}")
    print(f"  - Samples: {args.num_samples}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Device: {args.device}")
    
    print("\n[1/4] Generating data...")
    X_data, Y_data = generate_data(num_samples=args.num_samples, seed=args.seed)
    train_loader, test_loader = create_dataloaders(
        X_data, Y_data, 
        batch_size=args.batch_size, 
        train_split=0.8,
        seed=args.seed
    )
    print(f"  - Train samples: {len(train_loader.dataset)}")
    print(f"  - Test samples: {len(test_loader.dataset)}")
    
    print("\n[2/4] Initializing model...")
    model = HybridModel(
        input_dim=1,
        hidden_dims=[32, 64],
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        output_dim=4
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params}")
    print(f"  - Trainable parameters: {trainable_params}")
    
    print("\n[3/4] Training model...")
    print("-" * 60)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        device=args.device,
        verbose=True
    )
    
    print("-" * 60)
    print(f"\nBest Test Loss: {history['best_test_loss']:.4f}")
    
    print("\n[4/4] Detailed evaluation...")
    
    test_dataset = test_loader.dataset
    X_test = test_dataset.tensors[0]
    Y_test = test_dataset.tensors[1]
    
    eval_results = detailed_evaluation(model, X_test, Y_test, device=args.device)
    
    print(f"\nFinal Results:")
    print(f"  - Total MSE: {eval_results['total_mse']:.4f}")
    print(f"  - Per-target MSE:")
    for name, mse in eval_results['per_target_mse'].items():
        print(f"      {name}: {mse:.4f}")
    
    plot_path = os.path.join(args.results_dir, "training_curve.png")
    plot_training_history(history, save_path=plot_path)
    
    model_path = os.path.join(args.results_dir, "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'n_qubits': N_QUBITS,
            'n_layers': N_LAYERS,
            'hidden_dims': [32, 64],
        }
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    metrics_path = os.path.join(args.results_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Task 11: PQC Embedding Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Qubits: {N_QUBITS}\n")
        f.write(f"  PQC Layers: {N_LAYERS}\n")
        f.write(f"  MLP Hidden Dims: [32, 64]\n")
        f.write(f"  Samples: {args.num_samples}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning Rate: {args.lr}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Best Test Loss (MSE): {history['best_test_loss']:.4f}\n")
        f.write(f"  Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Test Loss: {history['test_loss'][-1]:.4f}\n\n")
        f.write(f"Per-Target MSE:\n")
        for name, mse in eval_results['per_target_mse'].items():
            f.write(f"  {name}: {mse:.4f}\n")
        f.write(f"\nBaseline Comparison:\n")
        f.write(f"  Baseline Loss: 0.5397\n")
        f.write(f"  Our Best Loss: {history['best_test_loss']:.4f}\n")
        improvement = (0.5397 - history['best_test_loss']) / 0.5397 * 100
        f.write(f"  Improvement: {improvement:.1f}%\n")
    print(f"Metrics saved to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return history, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 11: PQC Embedding")
    parser.add_argument("--num_samples", type=int, default=2048, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--scheduler_step", type=int, default=20, help="LR scheduler step size")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5, help="LR scheduler gamma")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    
    args = parser.parse_args()
    main(args)
