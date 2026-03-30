"""
Visualization utilities for QGNN circuits and results.
"""

import cirq
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os


def draw_circuit(
    circuit: cirq.Circuit,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8)
) -> str:
    """
    Draw a Cirq circuit and optionally save to file.
    
    Args:
        circuit: Cirq circuit to draw
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Text representation of circuit
    """
    # Get text representation
    circuit_text = str(circuit)
    
    # Create figure with circuit diagram
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.02, 0.98, circuit_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=8,
            verticalalignment='top')
    ax.axis('off')
    ax.set_title('QGNN Circuit Diagram', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Circuit diagram saved to {output_path}")
    
    plt.close()
    
    return circuit_text


def draw_circuit_svg(
    circuit: cirq.Circuit,
    output_path: str
) -> None:
    """
    Draw circuit using Cirq's SVG output.
    
    Args:
        circuit: Circuit to draw
        output_path: Path for SVG output
    """
    svg = cirq.contrib.svg.SVGCircuit(circuit)
    with open(output_path, 'w') as f:
        f.write(str(svg))
    print(f"Circuit SVG saved to {output_path}")


def plot_graph_structure(
    n_nodes: int,
    edge_index: np.ndarray,
    node_labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> None:
    """
    Plot the graph structure used for entanglement.
    
    Args:
        n_nodes: Number of nodes
        edge_index: (2, E) edge indices
        node_labels: Optional labels for nodes
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Arrange nodes in a circle
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Draw edges
    unique_edges = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        edge_key = (min(src, dst), max(src, dst))
        if edge_key not in unique_edges:
            unique_edges.add(edge_key)
            ax.plot([x[src], x[dst]], [y[src], y[dst]], 
                   'b-', alpha=0.5, linewidth=2)
    
    # Draw nodes
    ax.scatter(x, y, s=500, c='lightblue', edgecolors='darkblue', 
               linewidths=2, zorder=5)
    
    # Add labels
    if node_labels is None:
        node_labels = [f'q{i}' for i in range(n_nodes)]
    
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(node_labels[i], (xi, yi), ha='center', va='center',
                   fontsize=12, fontweight='bold')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Graph Structure for QGNN Entanglement\n({len(unique_edges)} edges)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Graph structure saved to {output_path}")
    
    plt.close()


def plot_training_history(
    history: dict,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Dictionary with 'loss', 'accuracy', etc.
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Train Acc')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {output_path}")
    
    plt.close()


def plot_qgnn_architecture(
    n_qubits: int = 8,
    n_layers: int = 3,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Create a schematic diagram of the QGNN architecture.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout parameters
    qubit_spacing = 1.0
    layer_spacing = 3.0
    gate_height = 0.3
    gate_width = 0.8
    
    # Draw qubits
    for i in range(n_qubits):
        y = -i * qubit_spacing
        ax.plot([0, (n_layers + 1) * layer_spacing], [y, y], 
               'k-', linewidth=1)
        ax.text(-0.5, y, f'q{i}', ha='right', va='center', fontsize=10)
    
    # Draw layers
    for layer in range(n_layers):
        x_base = (layer + 0.5) * layer_spacing
        
        # Encoding gates (blue)
        for i in range(n_qubits):
            y = -i * qubit_spacing
            rect = plt.Rectangle((x_base - gate_width/2, y - gate_height/2),
                                 gate_width, gate_height, 
                                 facecolor='lightblue', edgecolor='blue')
            ax.add_patch(rect)
            ax.text(x_base, y, 'Enc', ha='center', va='center', fontsize=8)
        
        # Variational gates (green)
        x_var = x_base + 1.0
        for i in range(n_qubits):
            y = -i * qubit_spacing
            rect = plt.Rectangle((x_var - gate_width/2, y - gate_height/2),
                                 gate_width, gate_height,
                                 facecolor='lightgreen', edgecolor='green')
            ax.add_patch(rect)
            ax.text(x_var, y, 'Var', ha='center', va='center', fontsize=8)
        
        # Entanglement (CZ gates) - show some example connections
        x_ent = x_var + 1.0
        example_edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
        for src, dst in example_edges:
            if src < n_qubits and dst < n_qubits:
                y1, y2 = -src * qubit_spacing, -dst * qubit_spacing
                ax.plot([x_ent, x_ent], [y1, y2], 'purple', linewidth=2)
                ax.scatter([x_ent, x_ent], [y1, y2], c='purple', s=50, zorder=5)
    
    # Measurements at the end
    x_meas = (n_layers + 0.5) * layer_spacing
    for i in range(n_qubits):
        y = -i * qubit_spacing
        ax.scatter([x_meas], [y], marker='s', s=100, c='red', zorder=5)
        ax.text(x_meas + 0.3, y, '⟨Z⟩', ha='left', va='center', fontsize=10)
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue', label='Encoding (Rx, Ry, Rz)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='green', label='Variational (Ry, Rz)'),
        plt.Line2D([0], [0], color='purple', linewidth=2, label='CZ Entanglement'),
        plt.Line2D([0], [0], marker='s', color='red', linestyle='', markersize=10, label='Z Measurement')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_xlim(-1, (n_layers + 1) * layer_spacing + 1)
    ax.set_ylim(-(n_qubits) * qubit_spacing, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'QGNN Architecture\n{n_qubits} qubits, {n_layers} layers', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Architecture diagram saved to {output_path}")
    
    plt.close()


def create_circuit_description(
    n_qubits: int = 8,
    n_layers: int = 3,
    n_features: int = 3
) -> str:
    """
    Create a detailed text description of the QGNN circuit.
    
    Returns:
        Formatted description string
    """
    desc = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QUANTUM GRAPH NEURAL NETWORK (QGNN)                       ║
║                         Circuit Architecture                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Configuration:                                                              ║
║    • Qubits: {n_qubits} (one per selected particle)                              ║
║    • Layers: {n_layers} (data re-uploading)                                        ║
║    • Features per particle: {n_features} (Δη, Δφ, log_pT)                          ║
║                                                                              ║
║  Layer Structure (repeated {n_layers}x):                                          ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │  1. FEATURE ENCODING                                                   │  ║
║  │     ├─ Rx(Δη_i): Encode relative rapidity                             │  ║
║  │     ├─ Ry(Δφ_i): Encode relative azimuthal angle                      │  ║
║  │     └─ Rz(log_pT_i): Encode log transverse momentum                   │  ║
║  │                                                                        │  ║
║  │  2. VARIATIONAL ROTATION (trainable)                                   │  ║
║  │     ├─ Ry(θ_i): Y-rotation with learned parameter                     │  ║
║  │     └─ Rz(φ_i): Z-rotation with learned parameter                     │  ║
║  │                                                                        │  ║
║  │  3. GRAPH ENTANGLEMENT                                                 │  ║
║  │     └─ CZ(q_i, q_j): Controlled-Z on k-NN edges                       │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  Measurement:                                                                ║
║    • Observable: Z-expectation on each qubit                                 ║
║    • Output: [{n_qubits} expectation values] → Classical MLP → Classification     ║
║                                                                              ║
║  Trainable Parameters: {n_layers * n_qubits * 2} (variational rotations)              ║
║                                                                              ║
║  Graph Structure:                                                            ║
║    • k-NN graph with k=4 in (η, φ) space                                    ║
║    • Entanglement topology reflects particle proximity                       ║
║    • Physically motivated: nearby particles are correlated                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    return desc
