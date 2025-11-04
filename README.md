# Adaptive-Variational-Quantum-Circuits-for-Resource-Efficient-Hybrid-Learning-in-NISQ-Environments

This project is about building Adaptive Variational Quantum Circuits (VQCs) — hybrid quantum-classical machine learning models that can modify their own structure during training. Instead of keeping the circuit fixed, the adaptive VQC will decide when to add/remove layers, adjust entanglement, or deactivate qubits based on training performance (e.g., loss stagnation, gradient sensitivity).

The purpose is to make quantum machine learning models more resource-efficient for Noisy Intermediate-Scale Quantum (NISQ) devices, where qubits are limited and noisy.

The project will be implemented completely in software using simulators like Qiskit Aer or PennyLane, and will be tested on datasets such as MNIST (digits 0 vs 1) and Iris.

## Project Structure
```
adaptive-vqc/
├── core/                        # Quantum circuit architectures
│   ├── vqc_fixed.py             # Standard static VQC
│   ├── vqc_adaptive.py          # Adaptive VQC with dynamic layers
│   └── layers.py                # Gate templates (rotation, entanglement)
├── data/                        # Dataset loaders
│   ├── iris.py
│   ├── mnist_binary.py
│   └── synthetic_xor.py
├── training/                    # Training & adaptation logic
│   ├── trainer.py               # Training loop
│   ├── adaptive_controller.py   # Adaptivity rules (loss/gradient)
│   └── noise_models.py          # Qiskit Aer noise simulation
├── evaluation/                  # Metrics & visualization
│   ├── metrics.py
│   ├── plot_curves.py
│   ├── plot_resources.py
│   └── visualize_circuit.py
├── config/                      # Experiment configs (JSON/YAML)
│   ├── mnist_fixed.json
│   ├── mnist_adaptive.json
│   └── iris_adaptive.json
├── results/                     # Logs & outputs
│   ├── logs/
│   ├── plots/
│   └── models/
├── scripts/                     # Run experiments from CLI
│   ├── train.py
│   ├── eval.py
│   └── run_noise_sim.py
├── notebooks/                   # Jupyter notebooks for demos
│   └── demo_mnist.ipynb
├── paper/                       # (Optional) write-up/report
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
└── LICENSE
```

#  **How to Run the Experiments**

---

##  **A. Run Fixed VQC**

> ###  Command
> ```bash
> python scripts/train.py --config config/iris_fixed.json --verbose
> ```
>
> ###  Description
> - Loads dataset and configuration.  
> - Trains a static VQC.  
> - Saves metrics and plots to `results/test_run/`.

---

##  **B. Run Adaptive VQC**

> ###  Command
> ```bash
> python scripts/train_adaptive.py --config config/iris_adaptive.json --verbose
> ```
>
> ###  Description
> - Uses the same dataset and hyperparameters.  
> - Dynamically modifies the circuit structure based on performance plateaus.  
> - Saves results to `results/adaptive_run/`.

---

##  **C. Compare Both Experiments**

> ### 💻 Command
> ```bash
> python scripts/compare_results.py \
> --adaptive results/adaptive_run/results.json \
> --fixed results/test_run/results.json
> ```
>
> ###  Generates
> - `comparison_summary.json` — summarized metrics (accuracy, loss, layers, parameters).  
> - `comparison_plot.png` — visual side-by-side bar chart.

---


##Implementation Steps

Phase 1: Core Prototype

Implement a fixed-depth VQC in core/vqc_fixed.py

Train on Iris/MNIST (binary) using training/trainer.py

Add a basic adaptive controller that adds layers if loss plateaus

Save training logs (CSV) and accuracy plots

Phase 2: Adaptive Strategy Suite

Extend adaptivity with:

Gradient-based pruning

Resource budget limits (max gates/qubits)

Add JSON/YAML configs for flexible experiments

Create plots: accuracy vs depth, gate usage vs time

Phase 3: NISQ Simulation & Noise

Add Qiskit Aer noise models (depolarizing, amplitude damping, phase damping)

Run both adaptive & fixed circuits under noise

(Optional) Test on IBM Q backend if API access is available

Build a CLI dashboard to switch datasets/circuits easily

Phase 4: Benchmarking & Final Polish

Run benchmarks across all configs (4–5 variants)

Generate reports with plots: resource–accuracy trade-offs

Clean up repo:

README.md with usage examples

Demo notebook (demo_mnist.ipynb)

Experiment logs in /results/

(Optional) Export a short write-up in /paper/

##Deliverables in GitHub

Modular code (core, training, evaluation)

Config-driven experiment setup

Logs + CSV files for reproducibility

Visualizations of accuracy, depth, and resource usage

Noise-aware simulation results

A user-friendly README with examples



##Outcome
By the end, the GitHub repo will:
1)Show how adaptive VQCs balance performance and resource efficiency
2)Provide reusable training pipelines for other quantum or classical models
3)Demonstrate results through plots and benchmarks
4)Be structured well enough for future extensions (new datasets, new adaptive strategies, or even non-quantum AI models)













