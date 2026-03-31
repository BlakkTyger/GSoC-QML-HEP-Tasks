"""
Task 6: Quantum Representation Learning
Quantum Similarity Network using SWAP test for contrastive learning on MNIST.

Architecture:
- Preprocessing: 4 quadrant means from 28x28 images
- Trainable encoding: theta = params[i,0] * x[i] + params[i,1]
- Separate parameters for img1 (RY) and img2 (RX)
- SWAP test for fidelity measurement
- Contrastive loss for similarity learning
- All 10 MNIST digit classes
"""

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Configuration
n_qubits_per_state = 4
total_wires = 1 + 2 * n_qubits_per_state  # 1 ancilla + 4 + 4 = 9 qubits

# Create quantum device
dev = qml.device("default.qubit", wires=total_wires)


def preprocess_image(img):
    """
    Preprocess image by computing mean of 4 quadrants.
    This preserves more spatial information than simple resize.
    
    Args:
        img: 28x28 tensor image
    
    Returns:
        4-dimensional tensor with quadrant means
    """
    if isinstance(img, torch.Tensor):
        img_np = img.numpy()
    else:
        img_np = img
    
    # Compute mean of each 14x14 quadrant
    q1 = np.mean(img_np[:14, :14])   # Top-left
    q2 = np.mean(img_np[:14, 14:])   # Top-right
    q3 = np.mean(img_np[14:, :14])   # Bottom-left
    q4 = np.mean(img_np[14:, 14:])   # Bottom-right
    
    return torch.tensor([q1, q2, q3, q4], dtype=torch.float32)


def encode_image_ry(image, params, wires):
    """Encode using RY rotations for the first image."""
    for i, wire in enumerate(wires):
        theta = params[i, 0] * image[i] + params[i, 1]
        qml.RY(theta, wires=wire)


def encode_image_rx(image, params, wires):
    """Encode using RX rotations for the second image."""
    for i, wire in enumerate(wires):
        theta = params[i, 0] * image[i] + params[i, 1]
        qml.RX(theta, wires=wire)


@qml.qnode(dev, interface="torch")
def quantum_circuit(image1, image2, params1, params2):
    """
    SWAP test circuit for measuring similarity between two images.
    
    Args:
        image1: First preprocessed image (4 features)
        image2: Second preprocessed image (4 features)
        params1: Parameters for first image encoding
        params2: Parameters for second image encoding
    
    Returns:
        Expectation value of PauliZ on ancilla (fidelity measure)
    """
    # Encode first image with RY rotations on wires 1 to n_qubits_per_state
    encode_image_ry(image1, params1, wires=list(range(1, n_qubits_per_state + 1)))
    
    # Encode second image with RX rotations on wires n_qubits_per_state+1 to total_wires
    encode_image_rx(image2, params2, wires=list(range(n_qubits_per_state + 1, total_wires)))
    
    # Perform SWAP test
    qml.Hadamard(wires=0)
    for i in range(n_qubits_per_state):
        qml.CSWAP(wires=[0, 1 + i, n_qubits_per_state + 1 + i])
    qml.Hadamard(wires=0)
    
    return qml.expval(qml.PauliZ(0))


class QuantumNet(torch.nn.Module):
    """
    Quantum Similarity Network using SWAP test.
    """
    def __init__(self):
        super(QuantumNet, self).__init__()
        # Separate trainable parameters for the two images
        self.params1 = torch.nn.Parameter(torch.randn(n_qubits_per_state, 2))
        self.params2 = torch.nn.Parameter(torch.randn(n_qubits_per_state, 2))
    
    def forward(self, img1, img2):
        proc_img1 = preprocess_image(img1)
        proc_img2 = preprocess_image(img2)
        fidelity = quantum_circuit(proc_img1, proc_img2, self.params1, self.params2)
        return fidelity


def contrastive_loss(fidelity, label):
    """
    Contrastive loss function.
    
    Args:
        fidelity: Quantum fidelity measure
        label: 1 if same class, 0 if different class
    
    Returns:
        Loss value
    """
    # Same class: maximize fidelity -> minimize (1 - fidelity)^2
    # Different class: minimize fidelity -> minimize fidelity^2
    return label * (1 - fidelity)**2 + (1 - label) * (fidelity)**2


def create_results_dir():
    """Create results directory."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def train(num_epochs=50, num_iterations=100, num_samples=4000, lr=0.02):
    """
    Train the Quantum Similarity Network.
    
    Args:
        num_epochs: Number of training epochs
        num_iterations: Iterations per epoch
        num_samples: Number of MNIST samples to use
        lr: Learning rate
    """
    print("=" * 60)
    print("Task 6: Quantum Representation Learning (V2)")
    print("=" * 60)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load MNIST dataset
    print("\n[1/4] Loading MNIST Dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0))  # Remove channel dimension
    ])
    
    mnist_train = torchvision.datasets.MNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Load data samples
    data = [(mnist_train[i][0], mnist_train[i][1]) for i in range(num_samples)]
    print(f"Loaded {len(data)} training samples")
    print(f"Using ALL 10 MNIST classes")
    
    # Verify data loading
    print("\n[2/4] Verifying Data Loading...")
    sample_img, sample_label = data[0]
    print(f"Sample image shape: {sample_img.shape}")
    print(f"Sample label: {sample_label}")
    proc_sample = preprocess_image(sample_img)
    print(f"Preprocessed shape: {proc_sample.shape}")
    print(f"Preprocessed values: {proc_sample.numpy()}")
    
    # Initialize model
    print("\n[3/4] Initializing Quantum Model...")
    qnet = QuantumNet()
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    
    print(f"Total qubits: {total_wires}")
    print(f"Parameters per image: {n_qubits_per_state * 2}")
    print(f"Total parameters: {n_qubits_per_state * 2 * 2}")
    print(f"Learning rate: {lr}")
    
    # Training loop
    print("\n[4/4] Training...")
    print("-" * 60)
    
    history = {
        'loss': [],
        'accuracy': [],
        'fidelity_same': [],
        'fidelity_diff': []
    }
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        fidelities_same = []
        fidelities_diff = []
        
        for _ in range(num_iterations):
            # Sample random pair
            idx1, idx2 = np.random.randint(0, len(data)), np.random.randint(0, len(data))
            img1, label1 = data[idx1]
            img2, label2 = data[idx2]
            
            # Ground truth: 1 if same class, 0 if different
            target = 1.0 if label1 == label2 else 0.0
            
            # Forward pass
            optimizer.zero_grad()
            fidelity = qnet(img1, img2)
            loss = contrastive_loss(fidelity, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track fidelities
            fid_val = fidelity.item()
            if target == 1.0:
                fidelities_same.append(fid_val)
            else:
                fidelities_diff.append(fid_val)
            
            # Compute accuracy
            predicted_label = 1.0 if fidelity >= 0.5 else 0.0
            if predicted_label == target:
                correct_predictions += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / num_iterations
        accuracy = correct_predictions / num_iterations * 100
        avg_fid_same = np.mean(fidelities_same) if fidelities_same else 0
        avg_fid_diff = np.mean(fidelities_diff) if fidelities_diff else 0
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        history['fidelity_same'].append(avg_fid_same)
        history['fidelity_diff'].append(avg_fid_diff)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%, "
                  f"Fid(same)={avg_fid_same:.3f}, Fid(diff)={avg_fid_diff:.3f}")
    
    print("-" * 60)
    print(f"Training Complete!")
    print(f"Final Accuracy: {history['accuracy'][-1]:.1f}%")
    
    return qnet, history, data


def evaluate(qnet, data, num_pairs=200):
    """
    Evaluate the trained model.
    
    Args:
        qnet: Trained QuantumNet
        data: Test data
        num_pairs: Number of pairs to evaluate
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating on test pairs...")
    
    correct = 0
    fidelities_same = []
    fidelities_diff = []
    
    with torch.no_grad():
        for _ in range(num_pairs):
            idx1, idx2 = np.random.randint(0, len(data)), np.random.randint(0, len(data))
            img1, label1 = data[idx1]
            img2, label2 = data[idx2]
            
            target = 1.0 if label1 == label2 else 0.0
            fidelity = qnet(img1, img2).item()
            
            if target == 1.0:
                fidelities_same.append(fidelity)
            else:
                fidelities_diff.append(fidelity)
            
            predicted = 1.0 if fidelity >= 0.5 else 0.0
            if predicted == target:
                correct += 1
    
    metrics = {
        'accuracy': correct / num_pairs * 100,
        'avg_fidelity_same': np.mean(fidelities_same),
        'avg_fidelity_diff': np.mean(fidelities_diff),
        'std_fidelity_same': np.std(fidelities_same),
        'std_fidelity_diff': np.std(fidelities_diff),
        'fidelity_gap': np.mean(fidelities_same) - np.mean(fidelities_diff)
    }
    
    return metrics, fidelities_same, fidelities_diff


def plot_results(history, fidelities_same, fidelities_diff, save_dir):
    """Plot training history and fidelity distribution."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['accuracy'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 100])
    
    # Fidelity over epochs
    axes[1, 0].plot(epochs, history['fidelity_same'], 'g-', linewidth=2, label='Same Class')
    axes[1, 0].plot(epochs, history['fidelity_diff'], 'r-', linewidth=2, label='Different Class')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Fidelity')
    axes[1, 0].set_title('Average Fidelity During Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fidelity distribution
    bins = np.linspace(0, 1, 21)
    axes[1, 1].hist(fidelities_same, bins=bins, alpha=0.6, color='green',
                    label=f'Same Class (μ={np.mean(fidelities_same):.3f})')
    axes[1, 1].hist(fidelities_diff, bins=bins, alpha=0.6, color='red',
                    label=f'Different Class (μ={np.mean(fidelities_diff):.3f})')
    axes[1, 1].axvline(0.5, color='black', linestyle='--', label='Threshold')
    axes[1, 1].set_xlabel('Fidelity')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Fidelity Distribution (Test)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results_v2.png'), dpi=150)
    print(f"Plots saved to {save_dir}/training_results_v2.png")
    plt.show()


def main():
    """Main function."""
    results_dir = create_results_dir()
    
    # Train model
    qnet, history, data = train(
        num_epochs=50,
        num_iterations=100,
        num_samples=4000,
        lr=0.02
    )
    
    # Load test data for evaluation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0))
    ])
    mnist_test = torchvision.datasets.MNIST(
        root="./data", 
        train=False, 
        download=True, 
        transform=transform
    )
    test_data = [(mnist_test[i][0], mnist_test[i][1]) for i in range(1000)]
    
    # Evaluate
    metrics, fid_same, fid_diff = evaluate(qnet, test_data, num_pairs=200)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {metrics['accuracy']:.1f}%")
    print(f"Avg Fidelity (Same Class): {metrics['avg_fidelity_same']:.4f} ± {metrics['std_fidelity_same']:.4f}")
    print(f"Avg Fidelity (Diff Class): {metrics['avg_fidelity_diff']:.4f} ± {metrics['std_fidelity_diff']:.4f}")
    print(f"Fidelity Gap: {metrics['fidelity_gap']:.4f}")
    
    # Plot results
    plot_results(history, fid_same, fid_diff, results_dir)
    
    # Save metrics
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write("Task 6: Quantum Representation Learning - Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Classes: All 10 MNIST digits\n")
        f.write(f"  Qubits: {total_wires} (1 ancilla + 4 + 4)\n")
        f.write(f"  Parameters: {n_qubits_per_state * 2 * 2}\n")
        f.write(f"  Training samples: 4000\n")
        f.write(f"  Epochs: 50\n")
        f.write(f"  Learning Rate: 0.02\n\n")
        f.write("Test Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\nResults saved to {results_dir}/")
    
    # Save model
    torch.save({
        'params1': qnet.params1.data,
        'params2': qnet.params2.data
    }, os.path.join(results_dir, 'model.pt'))
    
    return metrics


if __name__ == "__main__":
    metrics = main()
