"""Training and evaluation utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
import matplotlib.pyplot as plt


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu"
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: The hybrid model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
) -> float:
    """
    Evaluate the model.
    
    Args:
        model: The hybrid model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to use
    
    Returns:
        Average loss on the dataset
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = model(x_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.005,
    scheduler_step: int = 20,
    scheduler_gamma: float = 0.5,
    device: str = "cpu",
    verbose: bool = True
) -> dict:
    """
    Train the model with learning rate scheduling.
    
    Args:
        model: The hybrid model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        scheduler_step: Step size for LR scheduler
        scheduler_gamma: Multiplicative factor for LR decay
        device: Device to use
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )
    criterion = nn.MSELoss()
    
    history = {
        "train_loss": [],
        "test_loss": [],
        "learning_rate": []
    }
    
    best_test_loss = float("inf")
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["learning_rate"].append(current_lr)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        scheduler.step()
        
        if verbose:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"LR: {current_lr:.6f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history["best_test_loss"] = best_test_loss
    
    return history


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(history["test_loss"], label="Test Loss", color="orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training and Test Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history["learning_rate"], color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    plt.close()


def detailed_evaluation(
    model: nn.Module,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    device: str = "cpu"
) -> dict:
    """
    Perform detailed evaluation with per-target metrics.
    
    Args:
        model: The hybrid model
        X_test: Test input tensor
        Y_test: Test target tensor
        device: Device to use
    
    Returns:
        Dictionary containing detailed metrics
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        predictions = model(X_test)
        
        mse_total = nn.MSELoss()(predictions, Y_test).item()
        
        target_names = ["x", "sin(x)", "cos(x)", "x²"]
        per_target_mse = {}
        
        for i, name in enumerate(target_names):
            mse = nn.MSELoss()(predictions[:, i], Y_test[:, i]).item()
            per_target_mse[name] = mse
    
    return {
        "total_mse": mse_total,
        "per_target_mse": per_target_mse,
        "predictions": predictions.cpu(),
        "targets": Y_test.cpu()
    }
