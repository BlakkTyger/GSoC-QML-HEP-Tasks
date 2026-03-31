"""Training and Evaluation Pipeline for KAN."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from tqdm import tqdm


class Trainer:
    """Trainer class for Kolmogorov-Arnold Network."""
    
    def __init__(
        self,
        model,
        device,
        learning_rate=3e-4,
        weight_decay=1e-5,
        epochs=15,
        scheduler_type='cosine'
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.scheduler = None
            
        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc': None,
            'lr': []
        }
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader, desc='Evaluating'):
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in tqdm(loader, desc=desc, leave=False):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader=None):
        """Full training loop."""
        print(f"\nStarting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, desc='Validation')
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                
                val_str = f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
            else:
                val_str = ""
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch:2d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                  f"{val_str} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with validation accuracy: {best_val_acc*100:.2f}%")
        
        print("-" * 60)
        return self.history
    
    def test(self, test_loader):
        """Final evaluation on test set."""
        test_loss, test_acc = self.evaluate(test_loader, desc='Testing')
        self.history['test_acc'] = test_acc
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        return test_acc
    
    @torch.no_grad()
    def get_predictions(self, loader):
        """Get predictions and true labels for confusion matrix."""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        for data, target in loader:
            data = data.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1).cpu()
            all_preds.append(pred)
            all_targets.append(target)
        
        return torch.cat(all_preds), torch.cat(all_targets)
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")
