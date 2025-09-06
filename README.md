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
