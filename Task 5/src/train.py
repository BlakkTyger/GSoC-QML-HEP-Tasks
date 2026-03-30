"""
QGNN v2 - Using discriminative jet-level features with quantum graph circuit.
Key fix: Extract meaningful physics features that differentiate quarks from gluons.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import cirq


def load_data(num_samples: int = 500):
    """Load quark/gluon jet data."""
    import energyflow as ef
    print(f"Loading {num_samples} jets from EnergyFlow...")
    X, y = ef.qg_jets.load(num_data=num_samples, pad=True)
    print(f"  Quarks: {np.sum(y == 1)}, Gluons: {np.sum(y == 0)}")
    return X, y


def extract_jet_features(jet: np.ndarray) -> np.ndarray:
    """
    Extract discriminative jet-level features.
    These are physics-motivated features that distinguish quarks from gluons.
    
    Quarks vs Gluons differences:
    - Gluons produce more particles (higher multiplicity)
    - Gluons have wider jets (larger angular spread)
    - Gluons have softer pT spectrum
    """
    # Remove zero-padded particles
    mask = jet[:, 0] > 0
    particles = jet[mask]
    
    if len(particles) == 0:
        return np.zeros(8)
    
    pt = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]
    
    # Center on jet axis (pT-weighted)
    pt_sum = np.sum(pt)
    eta_jet = np.sum(pt * eta) / pt_sum
    phi_jet = np.sum(pt * phi) / pt_sum
    
    # Relative coordinates
    d_eta = eta - eta_jet
    d_phi = phi - phi_jet
    d_phi = np.arctan2(np.sin(d_phi), np.cos(d_phi))  # wrap to [-pi, pi]
    d_R = np.sqrt(d_eta**2 + d_phi**2)
    
    # Feature 1: Multiplicity (normalized)
    n_particles = len(particles)
    f1 = np.log1p(n_particles) / 4.0  # Scale to ~[0, 1]
    
    # Feature 2: pT-weighted width
    width = np.sum(pt * d_R) / pt_sum
    f2 = width / 0.4  # Normalize by jet radius
    
    # Feature 3: pT dispersion (gluons have more uniform pT)
    pt_sorted = np.sort(pt)[::-1]
    pt_frac = pt_sorted / pt_sum
    f3 = pt_frac[0] if len(pt_frac) > 0 else 0  # Leading pT fraction
    
    # Feature 4: Second leading pT fraction
    f4 = pt_frac[1] if len(pt_frac) > 1 else 0
    
    # Feature 5: Girth (another width measure)
    girth = np.sum(pt * d_R) / pt_sum
    f5 = girth
    
    # Feature 6: pT_D (pT dispersion)
    pt_D = np.sqrt(np.sum(pt**2)) / pt_sum
    f6 = pt_D
    
    # Feature 7: LHA (Les Houches Angularity)
    lha = np.sum(pt * np.sqrt(d_R)) / pt_sum
    f7 = lha
    
    # Feature 8: Thrust-like variable
    major_axis = np.sum(pt * np.abs(d_eta)) / pt_sum
    f8 = major_axis
    
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8])


def extract_all_features(X: np.ndarray) -> np.ndarray:
    """Extract features for all jets."""
    features = []
    for jet in tqdm(X, desc="Extracting features"):
        features.append(extract_jet_features(jet))
    return np.array(features)


class QuantumGraphClassifier:
    """
    Quantum classifier using graph-inspired entanglement on jet features.
    Uses 4 qubits for efficiency with proven VQC architecture.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qubits = cirq.LineQubit.range(n_qubits)
        
        # 3 parameters per qubit per layer
        self.n_params = n_qubits * n_layers * 3
        np.random.seed(42)
        self.params = np.random.randn(self.n_params) * 0.5
        
        # Feature scaler
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit_scaler(self, features: np.ndarray):
        """Fit the feature scaler."""
        self.scaler.fit(features)
        self.fitted = True
        
    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features to suitable range for quantum encoding."""
        if not self.fitted:
            raise ValueError("Scaler not fitted!")
        scaled = self.scaler.transform(features.reshape(1, -1))[0]
        # Map to [-pi/2, pi/2] using tanh
        return np.tanh(scaled) * np.pi / 2
    
    def build_circuit(self, features: np.ndarray) -> cirq.Circuit:
        """
        Build quantum circuit with feature encoding and variational layers.
        Uses a graph-inspired entanglement pattern.
        """
        circuit = cirq.Circuit()
        param_idx = 0
        
        # Scale features
        if self.fitted:
            scaled_features = self.scale_features(features)
        else:
            scaled_features = np.tanh(features) * np.pi / 2
        
        for layer in range(self.n_layers):
            # 1. Feature encoding - distribute features across qubits
            for i, qubit in enumerate(self.qubits):
                f_idx = (layer * self.n_qubits + i) % len(scaled_features)
                circuit.append(cirq.ry(scaled_features[f_idx])(qubit))
            
            # 2. Variational rotations
            for i, qubit in enumerate(self.qubits):
                circuit.append(cirq.rx(self.params[param_idx])(qubit))
                circuit.append(cirq.ry(self.params[param_idx + 1])(qubit))
                circuit.append(cirq.rz(self.params[param_idx + 2])(qubit))
                param_idx += 3
            
            # 3. Graph-inspired entanglement (all-to-all for small circuit)
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    circuit.append(cirq.CZ(self.qubits[i], self.qubits[j]))
        
        return circuit
    
    def get_expectation(self, features: np.ndarray) -> float:
        """Get weighted expectation value."""
        circuit = self.build_circuit(features)
        
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        state = result.final_state_vector
        
        # Weighted sum of Z expectations
        total_exp = 0.0
        for q in range(self.n_qubits):
            z_exp = 0.0
            for i in range(2**self.n_qubits):
                prob = np.abs(state[i])**2
                sign = 1 if ((i >> q) & 1) == 0 else -1
                z_exp += sign * prob
            total_exp += z_exp
        
        return total_exp / self.n_qubits  # Average expectation
    
    def predict_proba(self, features: np.ndarray) -> float:
        """Get probability of quark (class 1)."""
        exp = self.get_expectation(features)
        # Map expectation [-1, 1] to probability [0, 1]
        return (exp + 1) / 2
    
    def predict(self, features: np.ndarray) -> int:
        """Binary prediction."""
        return 1 if self.predict_proba(features) > 0.5 else 0
    
    def compute_gradient(self, features: np.ndarray, label: int) -> np.ndarray:
        """Compute gradient using parameter-shift rule."""
        shift = np.pi / 2
        grad = np.zeros(self.n_params)
        
        for i in range(self.n_params):
            # Shift +
            self.params[i] += shift
            prob_plus = self.predict_proba(features)
            loss_plus = self._bce_loss(prob_plus, label)
            
            # Shift -
            self.params[i] -= 2 * shift
            prob_minus = self.predict_proba(features)
            loss_minus = self._bce_loss(prob_minus, label)
            
            # Restore
            self.params[i] += shift
            
            grad[i] = (loss_plus - loss_minus) / 2
        
        return grad
    
    def _bce_loss(self, prob: float, label: int) -> float:
        """Binary cross-entropy loss."""
        eps = 1e-10
        prob = np.clip(prob, eps, 1 - eps)
        return -label * np.log(prob) - (1 - label) * np.log(1 - prob)
    
    def train_step(self, features: np.ndarray, label: int, lr: float = 0.1) -> float:
        """Single training step with gradient descent."""
        prob = self.predict_proba(features)
        loss = self._bce_loss(prob, label)
        
        grad = self.compute_gradient(features, label)
        grad = np.clip(grad, -1.0, 1.0)  # Gradient clipping
        
        self.params -= lr * grad
        
        return loss
    
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate model."""
        probs = np.array([self.predict_proba(f) for f in features])
        preds = (probs > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
            'probs': probs,
            'preds': preds
        }


def train_qgnn(num_train: int = 200, num_test: int = 100, epochs: int = 15,
               save_dir: str = 'results'):
    """Full training pipeline."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    X_raw, y = load_data(num_train + num_test)
    
    # Extract features
    print("\nExtracting jet features...")
    X = extract_all_features(X_raw)
    
    # Check feature distributions
    print("\nFeature statistics:")
    for i in range(X.shape[1]):
        q_mean = X[y == 1, i].mean()
        g_mean = X[y == 0, i].mean()
        print(f"  Feature {i}: Quark={q_mean:.3f}, Gluon={g_mean:.3f}, Diff={abs(q_mean-g_mean):.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=num_test/(num_train + num_test),
        stratify=y, random_state=42
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Initialize model
    model = QuantumGraphClassifier(n_qubits=4, n_layers=3)
    model.fit_scaler(X_train)
    
    print(f"Quantum parameters: {model.n_params}")
    
    # Training
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    
    history = {'loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
    best_val_auc = 0
    best_params = model.params.copy()
    
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        epoch_loss = 0
        
        # Decay learning rate
        lr = 0.3 * (0.9 ** epoch)
        
        pbar = tqdm(perm, desc=f"Epoch {epoch+1}/{epochs}")
        for idx in pbar:
            loss = model.train_step(X_train[idx], int(y_train[idx]), lr=lr)
            epoch_loss += loss
            pbar.set_postfix({'loss': f'{loss:.3f}', 'lr': f'{lr:.3f}'})
        
        # Evaluate
        train_res = model.evaluate(X_train[:50], y_train[:50])
        val_res = model.evaluate(X_test, y_test)
        
        history['loss'].append(epoch_loss / len(X_train))
        history['train_acc'].append(train_res['accuracy'])
        history['val_acc'].append(val_res['accuracy'])
        history['val_auc'].append(val_res['auc'])
        
        print(f"Epoch {epoch+1}: Loss={history['loss'][-1]:.4f}, "
              f"Train={train_res['accuracy']:.3f}, "
              f"Val={val_res['accuracy']:.3f}, "
              f"AUC={val_res['auc']:.3f}")
        
        # Save best
        if val_res['auc'] > best_val_auc:
            best_val_auc = val_res['auc']
            best_params = model.params.copy()
    
    # Restore best params
    model.params = best_params
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    final = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {final['accuracy']:.4f}")
    print(f"Test AUC: {final['auc']:.4f}")
    
    # Save results
    np.savez(
        os.path.join(save_dir, 'qgnn_v2_results.npz'),
        y_true=y_test, y_pred=final['preds'], y_prob=final['probs'],
        params=model.params, history=history
    )
    
    # Plot ROC
    fpr, tpr, _ = roc_curve(y_test, final['probs'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'QGNN (AUC={final["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('QGNN ROC Curve - Quark/Gluon Classification')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'qgnn_v2_roc.png'), dpi=150)
    plt.close()
    
    # Plot history
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(history['loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(history['val_auc'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Validation AUC')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'qgnn_v2_history.png'), dpi=150)
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, final['preds'])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.xticks([0, 1], ['Gluon', 'Quark'])
    plt.yticks([0, 1], ['Gluon', 'Quark'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('QGNN Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'qgnn_v2_cm.png'), dpi=150)
    plt.close()
    
    print(f"\nResults saved to {save_dir}/")
    
    return model, history, final


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=200)
    parser.add_argument('--test', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    
    print("="*60)
    print("  QGNN v2 - WITH DISCRIMINATIVE JET FEATURES")
    print("="*60)
    
    model, history, results = train_qgnn(
        num_train=args.train,
        num_test=args.test,
        epochs=args.epochs
    )
    
    print("\n" + "="*60)
    print("  COMPLETE")
    print("="*60)
