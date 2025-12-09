# Planning-Enhanced LLM Reasoning Analysis

## Project Overview

This project investigates why planning helps large language models perform better on reasoning tasks by analyzing internal representations and causal mechanisms.

## Research Methods

- **Reasoning Strategy Comparison**: Direct vs Chain-of-Thought vs Plan-and-Execute
- **Internal Representation Analysis**: Hidden states, logits, stability metrics
- **Causal Intervention Experiments**: Verify causal role of planning

## Project Structure

```
├── config/                 # Configuration files
├── src/                    # Source code
│   ├── strategies/         # Reasoning strategy implementations
│   ├── datasets/          # Dataset processing
│   ├── models/            # Model management
│   ├── analysis/          # Representation analysis
│   ├── experiments/       # Experimental framework
│   └── utils/             # Utility functions
├── data/                  # Data storage
├── results/               # Experimental results
├── notebooks/             # Jupyter notebooks
└── tests/                 # Test files
```

## Quick Start

```bash
pip install -r requirements.txt
python src/experiments/run_experiment.py --config config/default.yaml
```