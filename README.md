# Reasoning Beyond Chain-of-Thought: A Latent Computational Mode in Large Language Models

## Project Overview

**Reasoning Beyond Chain-of-Thought** investigates whether multi-step reasoning in Large Language Models (LLMs) corresponds to a latent internal mechanism that can be selectively activated, and whether Chain-of-Thought (CoT) prompting is uniquely responsible for activating this mechanism.

This project implements a two-stage pipeline using **Sparse Autoencoders (SAEs)** to identify and intervene on latent features causally associated with reasoning behavior. Our findings suggest that multi-step reasoning reflects a latent capability inherent to the model, and CoT prompting is just one of several effective triggers.

## Key Contributions

- **Methodological**: A two-stage pipeline using SAEs to identify reasoning-related latent features and causally validate their role through targeted steering interventions.
- **Empirical**: Experiments across six model families (up to 70B) demonstrate that steering a single latent feature matches or exceeds CoT performance without explicit reasoning steps.
- **Mechanistic**: Evidence that this internal reasoning mode is triggered early in generation and can override prompt-level constraints (e.g., `no_think` instructions).

## Key Features

- **Reasoning Strategy Comparison**: Supports Direct, Chain-of-Thought (CoT), and Hint strategies.
- **Activation & Representation Analysis**: Extracts and analyzes activations from various model layers, with Sparse AutoEncoder (SAE) support to identify reasoning features.
- **Causal Intervention (Latent Steering)**:
  - Identify reasoning-relevant latent features using SAEs.
  - Perform steering experiments to trigger reasoning without CoT.
  - specific scripts for `anti-steering` to suppress reasoning.
- **Multi-Dataset Support**: Built-in loaders for BBH, GSM8K, GPQA, MMLU, and more.
- **Visualization & Evaluation**: Scripts for visualizing latent feature activations and evaluating steering performance.

## Directory Structure

```
├── config/           # Model configuration files (Llama, Qwen, Gemma, etc.)
├── data/             # Datasets (BBH, GPQA, GSM8K, MMLU, etc.)
├── results/          # Experimental results and intermediate files
├── scripts/          # Bash scripts for running experiments and analysis
├── src/
│   ├── analysis/     # Activation and representation analysis (SAE, LatentAnalyzer, etc.)
│   ├── my_datasets/  # Dataset loaders
│   ├── strategies/   # Reasoning strategy implementations (direct, cot, hint, etc.)
│   ├── utils/        # Utility functions and config
│   ├── run_experiment.py # Main experiment entry point
│   ├── runner.py     # Experiment runner and logic
│   ├── evaluator.py  # Strategy evaluation
│   └── visualize_*.py# Visualization scripts
└── requirements.txt  # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the main experiment script (example for Llama, Qwen, Gemma, etc.):

```bash
python src/run_experiment.py --config config/Llama70B.yaml --multi_gpu --device cuda:0
```

Or use provided bash scripts for baseline, analysis, or steering experiments:

```bash
bash scripts/run_baseline.sh
bash scripts/run_analysis.sh
bash scripts/steering.sh
```

## Paper

For detailed methodology, experiments, and analysis, please refer to the accompanying paper (see PDF attachment).

---