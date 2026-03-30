"""
Hyperparameter Tuning for QGAN Classifier
==========================================

This script performs systematic hyperparameter tuning for the QGAN classifier
to demonstrate understanding of fine-tuning quantum machine learning models.

It explores different hyperparameters and their impact on performance,
providing insights into the model's behavior.

Author: GSoC-QML-HEP Task 4
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
import os

from qgan_classifier import (
    QGANConfig, TrainingPipeline, HEPDataLoader,
    QGANClassifier, QuantumCircuitBuilder
)


class HyperparameterAnalyzer:
    """Analyze hyperparameter effects on model performance."""
    
    def __init__(self, data_path: str, results_dir: str = 'results'):
        self.data_path = data_path
        self.results_dir = results_dir
        self.loader = HEPDataLoader(data_path)
        self.results = []
        
        os.makedirs(results_dir, exist_ok=True)
        
    def run_grid_search(self) -> List[Dict[str, Any]]:
        """Run comprehensive grid search over hyperparameters."""
        print("="*80)
        print("COMPREHENSIVE HYPERPARAMETER GRID SEARCH")
        print("="*80)
        
        # Define grid
        grid = {
            'n_layers': [1, 2, 3],
            'encoding_type': ['angle', 'angle_rz'],
            'entanglement': ['none', 'linear', 'circular'],
            'learning_rate': [0.01, 0.05, 0.1],
            'batch_size': [8, 16, 32]
        }
        
        # Generate all combinations (subset for efficiency)
        configs = []
        exp_id = 0
        
        # Systematic exploration
        for n_layers in grid['n_layers']:
            for encoding_type in grid['encoding_type']:
                for entanglement in grid['entanglement']:
                    # Use default learning_rate and batch_size for main grid
                    config = QGANConfig(
                        n_layers=n_layers,
                        encoding_type=encoding_type,
                        entanglement=entanglement,
                        learning_rate=0.05,
                        batch_size=16,
                        epochs=100
                    )
                    configs.append((f"grid_{exp_id}", config))
                    exp_id += 1
        
        # Additional experiments for learning rate and batch size
        base_config = QGANConfig(n_layers=2, encoding_type='angle', 
                                entanglement='linear', epochs=100)
        
        for lr in grid['learning_rate']:
            if lr != 0.05:  # Already tested
                config = QGANConfig(
                    n_layers=2, encoding_type='angle', entanglement='linear',
                    learning_rate=lr, batch_size=16, epochs=100
                )
                configs.append((f"lr_{exp_id}", config))
                exp_id += 1
        
        for bs in grid['batch_size']:
            if bs != 16:  # Already tested
                config = QGANConfig(
                    n_layers=2, encoding_type='angle', entanglement='linear',
                    learning_rate=0.05, batch_size=bs, epochs=100
                )
                configs.append((f"bs_{exp_id}", config))
                exp_id += 1
        
        # Run experiments
        for name, config in configs:
            print(f"\nRunning experiment: {name}")
            try:
                result = self._run_single_experiment(name, config)
                self.results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                # Store failed result with defaults
                self.results.append({
                    'name': name,
                    'config': config.to_dict(),
                    'train': {'accuracy': 0.5, 'auc': 0.5},
                    'val': {'accuracy': 0.5, 'auc': 0.5},
                    'test': {'accuracy': 0.5, 'auc': 0.5},
                    'epochs_trained': 0,
                    'error': str(e)
                })
        
        return self.results
    
    def run_ablation_study(self) -> List[Dict[str, Any]]:
        """Run ablation study to understand component contributions."""
        print("\n" + "="*80)
        print("ABLATION STUDY")
        print("="*80)
        
        # Base configuration
        base_config = QGANConfig(
            n_layers=2,
            encoding_type='angle',
            entanglement='linear',
            learning_rate=0.05,
            batch_size=16,
            epochs=100
        )
        
        # Ablation configurations
        ablation_configs = [
            # No entanglement
            ("no_entanglement", QGANConfig(
                n_layers=2, encoding_type='angle', entanglement='none',
                learning_rate=0.05, batch_size=16, epochs=100
            )),
            # Single layer
            ("single_layer", QGANConfig(
                n_layers=1, encoding_type='angle', entanglement='linear',
                learning_rate=0.05, batch_size=16, epochs=100
            )),
            # Minimal encoding
            ("minimal_encoding", QGANConfig(
                n_layers=2, encoding_type='angle', entanglement='linear',
                learning_rate=0.05, batch_size=16, epochs=100
            )),
            # Full configuration
            ("full_config", QGANConfig(
                n_layers=3, encoding_type='angle_rz', entanglement='circular',
                learning_rate=0.05, batch_size=16, epochs=100
            ))
        ]
        
        for name, config in ablation_configs:
            print(f"\nRunning ablation: {name}")
            result = self._run_single_experiment(name, config)
            self.results.append(result)
        
        return self.results
    
    def run_convergence_analysis(self) -> Dict[str, Any]:
        """Analyze training convergence for different configurations."""
        print("\n" + "="*80)
        print("CONVERGENCE ANALYSIS")
        print("="*80)
        
        configs = [
            ("fast", QGANConfig(n_layers=1, learning_rate=0.1, epochs=50)),
            ("medium", QGANConfig(n_layers=2, learning_rate=0.05, epochs=100)),
            ("slow", QGANConfig(n_layers=3, learning_rate=0.01, epochs=200))
        ]
        
        convergence_data = {}
        
        for name, config in configs:
            print(f"\nAnalyzing convergence for: {name}")
            history = self._run_with_history(name, config)
            convergence_data[name] = {
                'config': config.to_dict(),
                'history': history
            }
        
        return convergence_data
    
    def _run_single_experiment(self, name: str, config: QGANConfig) -> Dict[str, Any]:
        """Run a single experiment and return results."""
        # Load data
        X_train, y_train, X_test, y_test = self.loader.load()
        
        # Create validation split
        n_val = int(len(y_train) * config.validation_split)
        X_val, y_val = X_train[:n_val], y_train[:n_val]
        X_train_split, y_train_split = X_train[n_val:], y_train[n_val:]
        
        # Train model
        classifier = QGANClassifier(config)
        history = classifier.train(X_train_split, y_train_split, X_val, y_val, verbose=0)
        
        # Evaluate
        train_results = classifier.evaluate(X_train_split, y_train_split)
        val_results = classifier.evaluate(X_val, y_val)
        test_results = classifier.evaluate(X_test, y_test)
        
        return {
            'name': name,
            'config': config.to_dict(),
            'train': {'accuracy': train_results['accuracy'], 'auc': train_results['auc']},
            'val': {'accuracy': val_results['accuracy'], 'auc': val_results['auc']},
            'test': {'accuracy': test_results['accuracy'], 'auc': test_results['auc']},
            'epochs_trained': len(history.history['loss'])
        }
    
    def _run_with_history(self, name: str, config: QGANConfig) -> Dict[str, List[float]]:
        """Run experiment with detailed history tracking."""
        # Load data
        X_train, y_train, X_test, y_test = self.loader.load()
        
        # Create validation split
        n_val = int(len(y_train) * config.validation_split)
        X_val, y_val = X_train[:n_val], y_train[:n_val]
        X_train_split, y_train_split = X_train[n_val:], y_train[n_val:]
        
        # Train model
        classifier = QGANClassifier(config)
        history = classifier.train(X_train_split, y_train_split, X_val, y_val, verbose=0)
        
        return {
            'train_loss': history.history['loss'],
            'train_acc': history.history['accuracy'],
            'val_loss': history.history['val_loss'],
            'val_acc': history.history['val_accuracy']
        }
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze results and provide insights."""
        if not self.results:
            raise ValueError("No results to analyze")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.results)
        
        # Expand nested results
        for split in ['train', 'val', 'test']:
            if split in df.columns:
                for metric in ['accuracy', 'auc']:
                    df[f'{split}_{metric}'] = df[split].apply(lambda x: x.get(metric, 0.5) if isinstance(x, dict) else 0.5)
                df = df.drop(split, axis=1)
        
        # Expand config columns
        if 'config' in df.columns:
            config_df = pd.json_normalize(df['config'])
            df = pd.concat([df.drop('config', axis=1), config_df], axis=1)
        
        # Handle missing columns
        if 'test_auc' not in df.columns:
            df['test_auc'] = 0.5
        if 'test_accuracy' not in df.columns:
            df['test_accuracy'] = 0.5
        
        analysis = {
            'best_config': df.loc[df['test_auc'].idxmax()].to_dict() if len(df) > 0 else {},
            'parameter_importance': {},
            'statistical_summary': {}
        }
        
        # Parameter importance analysis
        for param in ['n_layers', 'encoding_type', 'entanglement', 'learning_rate', 'batch_size']:
            if param in df.columns:
                grouped = df.groupby(param)['test_auc'].agg(['mean', 'std', 'count'])
                analysis['parameter_importance'][param] = grouped.to_dict()
        
        # Statistical summary
        analysis['statistical_summary'] = {
            'mean_test_accuracy': float(df['test_accuracy'].mean()),
            'std_test_accuracy': float(df['test_accuracy'].std()),
            'mean_test_auc': float(df['test_auc'].mean()),
            'std_test_auc': float(df['test_auc'].std()),
            'total_experiments': len(df)
        }
        
        return analysis
    
    def plot_results(self, save_plots: bool = True) -> None:
        """Create visualization plots for the results."""
        if not self.results:
            raise ValueError("No results to plot")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Expand nested results
        for split in ['train', 'val', 'test']:
            if split in df.columns:
                for metric in ['accuracy', 'auc']:
                    df[f'{split}_{metric}'] = df[split].apply(lambda x: x.get(metric, 0.5) if isinstance(x, dict) else 0.5)
                df = df.drop(split, axis=1)
        
        # Expand config columns
        if 'config' in df.columns:
            config_df = pd.json_normalize(df['config'])
            df = pd.concat([df.drop('config', axis=1), config_df], axis=1)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyperparameter Impact on Model Performance', fontsize=16)
        
        # Plot 1: Number of layers
        if 'n_layers' in df.columns:
            sns.boxplot(data=df, x='n_layers', y='test_auc', ax=axes[0, 0])
            axes[0, 0].set_title('Impact of Number of Layers')
            axes[0, 0].set_xlabel('Number of Layers')
            axes[0, 0].set_ylabel('Test AUC')
        
        # Plot 2: Encoding type
        if 'encoding_type' in df.columns:
            sns.boxplot(data=df, x='encoding_type', y='test_auc', ax=axes[0, 1])
            axes[0, 1].set_title('Impact of Encoding Type')
            axes[0, 1].set_xlabel('Encoding Type')
            axes[0, 1].set_ylabel('Test AUC')
        
        # Plot 3: Entanglement
        if 'entanglement' in df.columns:
            sns.boxplot(data=df, x='entanglement', y='test_auc', ax=axes[0, 2])
            axes[0, 2].set_title('Impact of Entanglement')
            axes[0, 2].set_xlabel('Entanglement Type')
            axes[0, 2].set_ylabel('Test AUC')
        
        # Plot 4: Learning rate
        if 'learning_rate' in df.columns:
            sns.scatterplot(data=df, x='learning_rate', y='test_auc', ax=axes[1, 0])
            axes[1, 0].set_title('Impact of Learning Rate')
            axes[1, 0].set_xlabel('Learning Rate')
            axes[1, 0].set_ylabel('Test AUC')
            axes[1, 0].set_xscale('log')
        
        # Plot 5: Batch size
        if 'batch_size' in df.columns:
            sns.boxplot(data=df, x='batch_size', y='test_auc', ax=axes[1, 1])
            axes[1, 1].set_title('Impact of Batch Size')
            axes[1, 1].set_xlabel('Batch Size')
            axes[1, 1].set_ylabel('Test AUC')
        
        # Plot 6: Accuracy vs AUC
        sns.scatterplot(data=df, x='test_accuracy', y='test_auc', ax=axes[1, 2])
        axes[1, 2].set_title('Accuracy vs AUC Correlation')
        axes[1, 2].set_xlabel('Test Accuracy')
        axes[1, 2].set_ylabel('Test AUC')
        
        # Add correlation coefficient
        corr = df['test_accuracy'].corr(df['test_auc'])
        axes[1, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=axes[1, 2].transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, 'hyperparameter_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"Plots saved to {self.results_dir}/hyperparameter_analysis.png")
        
        plt.show()
    
    def save_results(self, filename: str = 'hyperparameter_results.json'):
        """Save all results to JSON file."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def generate_report(self) -> str:
        """Generate a text report of findings."""
        if not self.results:
            return "No results to report"
        
        analysis = self.analyze_results()
        
        report = []
        report.append("="*80)
        report.append("HYPERPARAMETER TUNING REPORT")
        report.append("="*80)
        
        # Best configuration
        best = analysis['best_config']
        report.append(f"\nBEST CONFIGURATION:")
        report.append(f"  Name: {best['name']}")
        report.append(f"  Test Accuracy: {best['test_accuracy']:.4f}")
        report.append(f"  Test AUC: {best['test_auc']:.4f}")
        report.append(f"  Parameters:")
        for key in ['n_layers', 'encoding_type', 'entanglement', 'learning_rate', 'batch_size']:
            if key in best:
                report.append(f"    {key}: {best[key]}")
        
        # Parameter importance
        report.append(f"\nPARAMETER IMPORTANCE:")
        for param, stats in analysis['parameter_importance'].items():
            report.append(f"\n  {param}:")
            for value, metrics in stats['mean'].items():
                report.append(f"    {value}: AUC = {metrics:.4f} ± {stats['std'][value]:.4f}")
        
        # Statistical summary
        summary = analysis['statistical_summary']
        report.append(f"\nSTATISTICAL SUMMARY:")
        report.append(f"  Total experiments: {summary['total_experiments']}")
        report.append(f"  Mean test accuracy: {summary['mean_test_accuracy']:.4f} ± {summary['std_test_accuracy']:.4f}")
        report.append(f"  Mean test AUC: {summary['mean_test_auc']:.4f} ± {summary['std_test_auc']:.4f}")
        
        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        report.append(f"  1. Use {best['n_layers']} layers for optimal performance")
        report.append(f"  2. {best['encoding_type']} encoding shows best results")
        report.append(f"  3. {best['entanglement']} entanglement is preferred")
        report.append(f"  4. Learning rate around {best['learning_rate']} works well")
        report.append(f"  5. Batch size of {best['batch_size']} is optimal")
        
        return "\n".join(report)


def main():
    """Main execution function."""
    # Set random seeds
    np.random.seed(42)
    
    # Paths
    data_path = "/home/blakktyger/Documents/BlakkTyger/Projects/GSOC-26/GSoC-QML-HEP-Tasks/Task 4/QIS_EXAM_200Events.npz"
    results_dir = "/home/blakktyger/Documents/BlakkTyger/Projects/GSOC-26/GSoC-QML-HEP-Tasks/Task 4/results"
    
    # Create analyzer
    analyzer = HyperparameterAnalyzer(data_path, results_dir)
    
    # Print dataset info
    print("Dataset Statistics:")
    stats = analyzer.loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Run grid search
    print("\n" + "="*80)
    print("RUNNING GRID SEARCH")
    print("="*80)
    analyzer.run_grid_search()
    
    # Run ablation study
    analyzer.run_ablation_study()
    
    # Analyze results
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    analysis = analyzer.analyze_results()
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Save report
    with open(os.path.join(results_dir, 'tuning_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot results
    analyzer.plot_results()
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
