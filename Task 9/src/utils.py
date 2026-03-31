"""Utility Functions for KAN."""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(history, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    if history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, [a*100 for a in history['train_acc']], 'b-', label='Train')
    if history['val_acc']:
        axes[1].plot(epochs, [a*100 for a in history['val_acc']], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    if history['lr']:
        axes[2].plot(epochs, history['lr'], 'g-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()
    return fig


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def save_metrics(history, test_acc, model, save_path):
    """Save metrics to text file."""
    n_params = count_parameters(model)
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Kolmogorov-Arnold Network - MNIST Classification Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Parameters: {n_params:,}\n")
        f.write(f"Trainable Parameters: {n_params:,}\n\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Epochs: {len(history['train_loss'])}\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Train Accuracy: {history['train_acc'][-1]*100:.2f}%\n")
        
        if history['val_acc']:
            f.write(f"Final Val Loss: {history['val_loss'][-1]:.4f}\n")
            f.write(f"Final Val Accuracy: {history['val_acc'][-1]*100:.2f}%\n")
            f.write(f"Best Val Accuracy: {max(history['val_acc'])*100:.2f}%\n")
        f.write("\n")
        
        f.write("TEST RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
        
        f.write("BASELINE COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline (Task 9.ipynb): 96.40%\n")
        f.write(f"Our Model: {test_acc*100:.2f}%\n")
        diff = (test_acc - 0.9640) * 100
        f.write(f"Difference: {diff:+.2f}%\n")
        
    print(f"Metrics saved to {save_path}")


def get_classification_report(y_true, y_pred):
    """Generate classification report."""
    return classification_report(y_true, y_pred, digits=4)
