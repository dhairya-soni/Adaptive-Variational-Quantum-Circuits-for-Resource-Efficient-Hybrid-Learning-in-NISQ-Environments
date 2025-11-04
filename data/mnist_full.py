"""
Full MNIST dataset loader for large-scale quantum machine learning
Dataset size: ~1.7GB
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import logging

def load_mnist_full(binary_classes=[0, 1], n_components=4, max_samples=10000):
    """
    Load full MNIST dataset for quantum machine learning
    
    Args:
        binary_classes: Two digits to classify (e.g., [0, 1])
        n_components: Number of PCA components for quantum encoding
        max_samples: Maximum samples to use (for memory management)
    
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed data for quantum circuits
    """
    logging.info("Loading full MNIST dataset (~1.7GB)...")
    logging.info("This may take a few minutes on first download...")
    
    # Load full MNIST dataset
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        logging.info(f"MNIST loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logging.info(f"Dataset size in memory: {X.nbytes / (1024**3):.2f} GB")
        
    except Exception as e:
        logging.error(f"Failed to download MNIST: {e}")
        logging.info("Falling back to smaller sample...")
        # Fallback to smaller version if download fails
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data, digits.target
    
    # Filter for binary classification
    binary_mask = np.isin(y, binary_classes)
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    
    # Relabel classes to 0 and 1
    y_binary = (y_binary == binary_classes[1]).astype(int)
    
    logging.info(f"Binary classification: {binary_classes[0]} vs {binary_classes[1]}")
    logging.info(f"Filtered dataset: {X_binary.shape[0]} samples")
    
    # Limit samples if requested (for faster experimentation)
    if max_samples and X_binary.shape[0] > max_samples:
        indices = np.random.choice(X_binary.shape[0], max_samples, replace=False)
        X_binary = X_binary[indices]
        y_binary = y_binary[indices]
        logging.info(f"Limited to {max_samples} samples for faster training")
    
    # Normalize pixel values to [0, 1]
    X_binary = X_binary / 255.0
    
    # Apply PCA for dimensionality reduction (quantum circuits need few features)
    logging.info(f"Applying PCA: {X_binary.shape[1]} → {n_components} features")
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_binary)
    
    # Standardize features
    scaler = StandardScaler()
    X_reduced = scaler.fit_transform(X_reduced)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    logging.info(f"Final dataset shapes:")
    logging.info(f"  Training: {X_train.shape}")
    logging.info(f"  Test: {X_test.shape}")
    logging.info(f"  Features after PCA: {n_components}")
    logging.info(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return X_train, X_test, y_train, y_test

def visualize_mnist_samples(X, y, title="MNIST Samples", n_samples=10):
    """Visualize some MNIST samples"""
    plt.figure(figsize=(12, 4))
    for i in range(min(n_samples, len(X))):
        plt.subplot(2, 5, i+1)
        # Reshape back to 28x28 if it's flattened
        if X[i].shape[0] == 784:
            img = X[i].reshape(28, 28)
        else:
            # For PCA-reduced data, can't visualize as image
            plt.text(0.5, 0.5, f'Class: {y[i]}', ha='center', va='center')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title(f'Sample {i+1}')
            continue
            
        plt.imshow(img, cmap='gray')
        plt.title(f'Class: {y[i]}')
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return plt.gcf()

# Configuration for large dataset experiment
MNIST_CONFIG = {
    "dataset": "mnist_full",
    "binary_classes": [0, 1],  # Can change to [3, 8] or any two digits
    "pca_components": 4,       # Reduce to quantum-friendly size
    "max_samples": 5000,       # Limit for faster initial testing
    "description": "Full MNIST binary classification with PCA reduction"
}

if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)
    X_train, X_test, y_train, y_test = load_mnist_full()
    print(f"Successfully loaded MNIST: {X_train.shape} training samples")