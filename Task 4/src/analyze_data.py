"""
Script to analyze the QIS_EXAM_200Events.npz dataset structure
for the QGAN High Energy Physics classification task.
"""

import numpy as np

def analyze_npz_file(filepath):
    """Load and analyze the NPZ file structure."""
    print("=" * 60)
    print("QGAN HEP Dataset Analysis")
    print("=" * 60)
    
    # Load the NPZ file
    data = np.load(filepath, allow_pickle=True)
    
    # List all arrays in the file
    print("\n1. FILES IN NPZ ARCHIVE:")
    print("-" * 40)
    for key in data.files:
        print(f"   - {key}")
    
    # Analyze each array
    print("\n2. DETAILED ARRAY ANALYSIS:")
    print("-" * 40)
    
    for key in data.files:
        arr = data[key]
        print(f"\n   [{key}]")
        print(f"   Type: {type(arr)}")
        print(f"   Dtype: {arr.dtype}")
        print(f"   Shape: {arr.shape}")
        
        # If it's a structured array or has nested structure
        if arr.dtype == object:
            print(f"   Content type: Object array (likely dict or nested)")
            print(f"   First element type: {type(arr.item()) if arr.ndim == 0 else type(arr[0])}")
            
            # Try to explore the structure
            if arr.ndim == 0:
                item = arr.item()
            else:
                item = arr[0]
            
            if isinstance(item, dict):
                print(f"   Dictionary keys: {list(item.keys())}")
                for k, v in item.items():
                    if isinstance(v, np.ndarray):
                        print(f"      '{k}': shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"      '{k}': type={type(v)}")
        else:
            print(f"   Min: {arr.min():.6f}")
            print(f"   Max: {arr.max():.6f}")
            print(f"   Mean: {arr.mean():.6f}")
            print(f"   Std: {arr.std():.6f}")
    
    return data


def explore_training_test_structure(data):
    """Deep dive into training and test data structure."""
    print("\n" + "=" * 60)
    print("3. DEEP STRUCTURE EXPLORATION")
    print("=" * 60)
    
    for key in ['training_input', 'test_input']:
        if key in data.files:
            arr = data[key]
            print(f"\n[{key}]")
            
            # Handle object arrays
            if arr.dtype == object:
                if arr.ndim == 0:
                    content = arr.item()
                else:
                    content = arr
                
                if isinstance(content, dict):
                    print(f"   This is a dictionary with keys: {list(content.keys())}")
                    
                    for label_key, samples in content.items():
                        print(f"\n   Label '{label_key}':")
                        if isinstance(samples, np.ndarray):
                            print(f"      Shape: {samples.shape}")
                            print(f"      Dtype: {samples.dtype}")
                            print(f"      Number of samples: {samples.shape[0]}")
                            if len(samples.shape) > 1:
                                print(f"      Features per sample: {samples.shape[1]}")
                            print(f"      Sample range: [{samples.min():.4f}, {samples.max():.4f}]")
                            print(f"      Sample mean: {samples.mean():.4f}")
                            print(f"      Sample std: {samples.std():.4f}")
                            
                            # Show first sample
                            print(f"      First sample (first 10 values): {samples[0][:10] if len(samples.shape) > 1 else samples[0]}")


def analyze_feature_statistics(data):
    """Analyze feature statistics for signal vs background."""
    print("\n" + "=" * 60)
    print("4. FEATURE STATISTICS (Signal vs Background)")
    print("=" * 60)
    
    training = data['training_input'].item() if data['training_input'].ndim == 0 else data['training_input']
    
    if isinstance(training, dict):
        # Assuming keys are labels (0 for background, 1 for signal)
        keys = list(training.keys())
        print(f"\n   Labels found: {keys}")
        
        for label in keys:
            samples = training[label]
            print(f"\n   Label {label} ({'Signal' if label == 1 else 'Background'}):")
            print(f"      Number of samples: {samples.shape[0]}")
            
            if len(samples.shape) > 1:
                n_features = samples.shape[1]
                print(f"      Number of features: {n_features}")
                
                print("\n      Per-feature statistics:")
                print("      " + "-" * 50)
                for i in range(min(n_features, 10)):  # Show first 10 features
                    feat = samples[:, i]
                    print(f"      Feature {i}: min={feat.min():8.4f}, max={feat.max():8.4f}, "
                          f"mean={feat.mean():8.4f}, std={feat.std():8.4f}")
                
                if n_features > 10:
                    print(f"      ... and {n_features - 10} more features")


def prepare_combined_dataset(data):
    """Show how to prepare combined X, y arrays for ML."""
    print("\n" + "=" * 60)
    print("5. COMBINED DATASET PREPARATION")
    print("=" * 60)
    
    for split in ['training_input', 'test_input']:
        split_data = data[split].item() if data[split].ndim == 0 else data[split]
        
        if isinstance(split_data, dict):
            X_list = []
            y_list = []
            
            for label, samples in split_data.items():
                X_list.append(samples)
                y_list.append(np.full(samples.shape[0], label))
            
            X = np.vstack(X_list)
            y = np.concatenate(y_list)
            
            split_name = "Training" if "training" in split else "Test"
            print(f"\n   {split_name} Set:")
            print(f"      X shape: {X.shape}")
            print(f"      y shape: {y.shape}")
            print(f"      Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
            print(f"      Feature range: [{X.min():.4f}, {X.max():.4f}]")


if __name__ == "__main__":
    filepath = "/home/blakktyger/Documents/BlakkTyger/Projects/GSOC-26/GSoC-QML-HEP-Tasks/Task 4/QIS_EXAM_200Events.npz"
    
    # Load and analyze
    data = analyze_npz_file(filepath)
    
    # Explore structure
    explore_training_test_structure(data)
    
    # Feature statistics
    analyze_feature_statistics(data)
    
    # Show combined dataset preparation
    prepare_combined_dataset(data)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
