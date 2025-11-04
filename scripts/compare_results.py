#!/usr/bin/env python3
"""
Compare FIXED and ADAPTIVE VQC experiment results.
Generates unified metrics, comparison plots, and summary tables.
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def load_json(path):
    """Safely load JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        return json.load(f)

def normalize(value, base):
    """Normalize metrics for fair visual comparison."""
    return value / base if base != 0 else 0

def compare_results(adaptive, fixed):
    """Return aligned comparison dictionary."""
    return {
        "Train Accuracy": [adaptive["final_train_accuracy"], fixed["final_train_accuracy"]],
        "Test Accuracy": [adaptive["final_test_accuracy"], fixed["final_test_accuracy"]],
        "Final Loss": [adaptive["final_loss"], fixed["final_loss"]],
        "Layers": [adaptive["final_circuit"]["layers"], fixed["circuit_info"]["n_layers"]],
        "Parameters": [adaptive["final_circuit"]["parameters"], fixed["circuit_info"]["n_parameters"]],
        "Total Adaptations": [adaptive["total_adaptations"], 0]
    }

def plot_comparison(results, outdir="results/comparison"):
    """Generate comparison bar plots."""
    os.makedirs(outdir, exist_ok=True)
    labels = list(results.keys())
    adaptive_vals = [v[0] for v in results.values()]
    fixed_vals = [v[1] for v in results.values()]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, adaptive_vals, width, label="Adaptive VQC", color="royalblue")
    plt.bar(x + width/2, fixed_vals, width, label="Fixed VQC", color="tomato")
    plt.xticks(x, labels, rotation=25)
    plt.ylabel("Metric Values")
    plt.title("Adaptive vs Fixed VQC Comparison (Iris Dataset)")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    
    for i, v in enumerate(adaptive_vals):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(fixed_vals):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    
    path = os.path.join(outdir, "comparison_plot.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Comparison plot saved to {path}")

def save_summary(results, outdir="results/comparison"):
    """Save metrics summary as JSON."""
    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Summary JSON saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare Adaptive vs Fixed VQC results")
    parser.add_argument("--adaptive", required=True, help="Path to adaptive results.json")
    parser.add_argument("--fixed", required=True, help="Path to fixed results.json")
    args = parser.parse_args()

    adaptive = load_json(args.adaptive)
    fixed = load_json(args.fixed)
    
    comparison = compare_results(adaptive, fixed)
    save_summary(comparison)
    plot_comparison(comparison)

if __name__ == "__main__":
    main()
