"""
Training pipeline for GNN models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import os


class Trainer:
    """
    Trainer class for GNN jet classifiers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model: PyTorch model to train
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """
        Train for one epoch with detailed progress.
        
        Returns:
            Average loss and AUC for the epoch
        """
        self.model.train()
        total_loss = 0
        all_labels = []
        all_probs = []
        n_batches = len(train_loader)
        
        pbar = tqdm(train_loader, 
                    desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
                    leave=False,
                    ncols=100)
        
        for batch_idx, data in enumerate(pbar):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            out = self.model(data)
            if isinstance(out, tuple):
                out = out[0]
            
            loss = self.criterion(out, data.y.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            
            probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            labels = data.y.view(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
            
            # Update progress bar with running loss
            running_loss = total_loss / ((batch_idx + 1) * train_loader.batch_size)
            pbar.set_postfix({'loss': f'{running_loss:.4f}'})
        
        avg_loss = total_loss / len(train_loader.dataset)
        auc = roc_auc_score(all_labels, all_probs)
        
        return avg_loss, auc
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "Eval") -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate the model with progress bar.
        
        Returns:
            Average loss, AUC, true labels, and prediction probabilities
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_probs = []
        
        pbar = tqdm(loader, desc=f"[{desc}]", leave=False, ncols=100)
        
        for data in pbar:
            data = data.to(self.device)
            
            out = self.model(data)
            if isinstance(out, tuple):
                out = out[0]
            
            loss = self.criterion(out, data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
            
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            labels = data.y.view(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
        
        avg_loss = total_loss / len(loader.dataset)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        auc = roc_auc_score(all_labels, all_probs)
        
        return avg_loss, auc, all_labels, all_probs
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
        model_name: str = "model"
    ) -> Dict[str, List[float]]:
        """
        Full training loop with validation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_path: Directory to save best model
            model_name: Name for saved model file
            
        Returns:
            Training history dictionary
        """
        best_val_auc = 0
        patience_counter = 0
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print("-" * 80)
        
        for epoch in range(epochs):
            # Training with detailed progress
            train_loss, train_auc = self.train_epoch(train_loader, epoch, epochs)
            
            # Validation with progress
            val_loss, val_auc, _, _ = self.evaluate(val_loader, desc="Val")
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            self.scheduler.step(val_auc)
            
            # Print epoch summary
            lr = self.optimizer.param_groups[0]['lr']
            improved = "*" if val_auc > best_val_auc else ""
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f} {improved} | "
                  f"LR: {lr:.2e}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(save_path, f'{model_name}_best.pt')
                    )
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        print(f"\nBest validation AUC: {best_val_auc:.4f}")
        
        return self.history
    
    def load_best_model(self, save_path: str, model_name: str = "model"):
        """Load the best saved model."""
        path = os.path.join(save_path, f'{model_name}_best.pt')
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded best model from {path}")
