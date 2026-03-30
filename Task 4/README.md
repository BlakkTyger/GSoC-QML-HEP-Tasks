# Task IV: Quantum Generative Adversarial Network (QGAN)

You will explore how best to apply a quantum generative adversarial network (QGAN) to solve a High Energy Data analysis issue, more specifically, separating the signal events from the background events. You should use the Google Cirq and Tensorflow Quantum (TFQ) libraries for this task.

A set of input samples (simulated with Delphes) is provided in NumPy NPZ format. In the input file, there are only 100 samples for training and 100 samples for testing so it won't take much computing resources to accomplish this task. The signal events are labeled with 1 while the background events are labeled with 0. 

Be sure to show that you understand how to fine tune your machine learning model to improve the performance. The performance can be evaluated with classification accuracy or Area Under ROC Curve (AUC). 

## Implementation Overview

This task implements a **QGAN-inspired Variational Quantum Classifier (VQC)** for HEP signal/background classification. While a traditional GAN generates data, we use the discriminator-like quantum circuit as a classifier - the most suitable approach for this binary classification task.

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Experiments
```bash
# Default configuration
python run_experiment.py

# Best performing configuration
python run_experiment.py --config best

# Hyperparameter tuning
python run_experiment.py --tune
```

### 3. Analyze Data
```bash
python analyze_data.py
```

## Project Structure

```
Task 4/
├── qgan_classifier.py       # Main quantum classifier implementation
├── hyperparameter_tuning.py # Systematic hyperparameter optimization
├── run_experiment.py        # Easy experiment runner
├── inference.py             # Standalone inference & evaluation script
├── analyze_data.py          # Data exploration and statistics
├── requirements.txt         # Python dependencies
├── QGAN_PLANNING_DOCUMENT.md # Detailed planning and research
├── FINAL_DOCUMENTATION.md   # Complete results and analysis
├── results/                 # Saved experiment results
└── QIS_EXAM_200Events.npz   # Dataset (200 HEP events)
```

## Key Results

- **Best Test Accuracy**: 0.71 (71%)
- **Best Test AUC**: 0.79
- **Signal Efficiency**: 0.62
- **Background Rejection**: 0.78
- **Optimal Configuration**: 3 layers, angle encoding, circular entanglement
- **Parameter Efficiency**: Only 30 trainable quantum parameters

### Hyperparameter Tuning Insights
| Parameter | Best Value | Impact |
|-----------|------------|--------|
| Layers | 3 | Higher layers improve expressivity |
| Encoding | angle | Simple RY encoding sufficient |
| Entanglement | circular | Critical for performance (+16% AUC vs none) |
| Learning Rate | 0.05 | Stable convergence |
| Batch Size | 16 | Good gradient estimates |

## Architecture

The selected architecture is an enhanced VQC with:
1. **Angle Encoding**: RY(π·x) gates for data embedding
2. **Variational Layers**: Hardware-efficient ansatz with entanglement
3. **Measurement**: Z expectation value
4. **Classical Post-processing**: Linear layer with sigmoid

## Fine-tuning Demonstration

The implementation includes comprehensive hyperparameter tuning:
- Grid search over layers, encoding, entanglement, learning rate, batch size
- Ablation studies to understand component contributions
- Convergence analysis for optimization
- Visualization of hyperparameter effects

## Understanding the QGAN Approach

For this classification task, we use the **discriminator component** of a GAN architecture as our quantum classifier. This approach:
- Leverages quantum feature spaces for better separation
- Provides parameter efficiency for small datasets
- Demonstrates quantum ML capabilities in HEP context

A full QGAN (generator + discriminator) would be more suitable for data augmentation tasks.

## Dependencies

- tensorflow>=2.8.0
- tensorflow-quantum>=0.7.0
- cirq>=0.15.0
- scikit-learn>=1.0.0
- numpy, pandas, matplotlib, seaborn, sympy

## Performance Notes

- Quantum simulation is slower than classical methods
- True quantum advantage requires real quantum hardware
- Current implementation demonstrates feasibility and methodology
- Performance competitive with classical baselines on small dataset

## Future Extensions

1. **Data Augmentation**: Add quantum generator for synthetic data
2. **Real Hardware**: Deploy on actual quantum processors
3. **Advanced Encodings**: Problem-specific quantum feature maps
4. **Hybrid Models**: Combine with classical ML techniques