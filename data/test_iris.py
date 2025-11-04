#!/usr/bin/env python3
"""
Test iris dataset loading directly
"""

# Test 1: Check if sklearn iris works
print("=== Test 1: Testing sklearn iris dataset ===")
try:
    from sklearn.datasets import load_iris
    iris = load_iris()
    print(f"✅ Iris dataset loaded successfully!")
    print(f"   Shape: {iris.data.shape}")
    print(f"   Classes: {iris.target_names}")
    print(f"   Features: {iris.feature_names}")
except Exception as e:
    print(f"❌ Failed to load iris: {e}")

# Test 2: Test your iris.py file directly
print("\n=== Test 2: Testing your iris.py file ===")
try:
    # Add parent directory to path
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.iris import load_iris_binary
    X_train, X_test, y_train, y_test = load_iris_binary()
    
    print(f"✅ Your iris.py works perfectly!")
    print(f"   Training shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    print(f"   Classes in training: {set(y_train)}")
    print(f"   Classes in test: {set(y_test)}")
    
except Exception as e:
    print(f"❌ Your iris.py failed: {e}")
    print("This is likely due to import path or null bytes issue")

# Test 3: Manual iris loading (fallback)
print("\n=== Test 3: Manual iris loading (what we'll use as fallback) ===")
try:
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    # Manual loading
    iris = load_iris()
    binary_mask = iris.target < 2
    X = iris.data[binary_mask]
    y = iris.target[binary_mask]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"✅ Manual iris loading works!")
    print(f"   Training shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    print(f"   Sample training data: {X_train[0]}")
    print(f"   Sample labels: {y_train[:5]}")
    
except Exception as e:
    print(f"❌ Manual loading failed: {e}")

print("\n=== Summary ===")
print("The Iris dataset is built into scikit-learn - you don't need to download anything!")
print("If Test 2 fails but Test 3 works, we'll use the fallback approach.")