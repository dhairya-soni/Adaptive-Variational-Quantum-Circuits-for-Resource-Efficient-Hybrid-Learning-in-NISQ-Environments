#!/usr/bin/env python3
"""
Debug script to test imports one by one
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

print("Testing imports one by one...")

# Test each import individually
imports_to_test = [
    ("core.vqc_fixed", "create_fixed_vqc"),
    ("core.vqc_adaptive", "create_adaptive_vqc, AdaptationConfig"),
    ("training.trainer", "VQCTrainer, TrainingConfig, train_vqc"),
    ("training.noise_models", "create_noise_model"),
    ("data.mnist_binary", "load_mnist_binary"),
    ("data.iris", "load_iris_binary"),
    ("evaluation.metrics", "compute_detailed_metrics"),
    ("evaluation.plot_curves", "plot_training_history"),
    ("evaluation.visualize_circuit", "plot_circuit_evolution")
]

for module_name, imports in imports_to_test:
    try:
        exec(f"from {module_name} import {imports}")
        print(f"✅ SUCCESS: {module_name}")
    except Exception as e:
        print(f"❌ FAILED: {module_name} - {e}")

print("\nDone testing imports!")