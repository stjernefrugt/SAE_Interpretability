# Sparse Autoencoders for Mechanistic Interpretability

**From X-ray Signal Decomposition to Neural Network Feature Extraction**

## Overview

This project trains a Sparse Autoencoder (SAE) on GPT-2 small activations to find interpretable computational features — and compares this with SVD decomposition, the standard tool I use for analyzing time-resolved X-ray scattering data at SLAC.

The core question: **How do we decompose superposed signals into interpretable components?**

In ultrafast X-ray scattering, I decompose measured signals (which are superpositions of structural/dynamical modes) using SVD to extract orthogonal temporal and structural components. This works because physical modes are few and approximately orthogonal.

Neural network activations present a harder version of the same problem: features are **numerous** (far more features than dimensions), **non-orthogonal** (features share directions via superposition), and **sparse** (only a few active at any time). SVD cannot recover them — we need Sparse Autoencoders.

## Key Results

| Method | Constraint | # Features | Interpretability |
|--------|-----------|------------|------------------|
| SVD | Orthogonal, ranked by variance | ≤ d_model (768) | Polysemantic — each component mixes concepts |
| SAE | Sparse, overcomplete | 8 × d_model (6144) | Monosemantic — each feature = one concept |

**Finding:** SAE features are individually interpretable (e.g., "Python code", "first-person narrative", "numbers/math"), while SVD components mix multiple unrelated concepts — exactly as predicted by the superposition hypothesis.

## Connection to X-ray Scattering Analysis

| Concept | X-ray Scattering | Neural Networks |
|---------|------------------|-----------------|
| Raw signal | Scattering pattern S(q,t) | Activation vector a ∈ ℝ^768 |
| Superposition | Modes overlap in measured signal | Features share directions in activation space |
| Decomposition tool | SVD: D = UΣV^T | SAE: a ≈ Σ_i f_i · d_i (sparse) |
| Components | Orthogonal structural modes | Sparse interpretable features |
| Why it works | Physical modes ≈ orthogonal, few | Computational features = sparse, many |

## Project Structure

```
├── sae/
│   ├── model.py          # SAE architecture (from scratch, ~100 lines)
│   ├── activations.py    # Activation collection via TransformerLens
│   └── analysis.py       # Feature interpretation + SVD comparison
├── notebooks/
│   ├── 01_collect_activations.ipynb   # Collect GPT-2 layer 6 activations
│   ├── 02_train_sae.ipynb             # Train SAE, monitor convergence
│   └── 03_interpret_features.ipynb    # SVD vs SAE feature analysis
└── results/                           # Trained models and figures
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA-capable GPU (tested on RTX 5070, 8GB VRAM).

## Method

1. **Collect activations**: Run GPT-2 small on 2M tokens from The Pile, cache residual stream activations at layer 6 (d=768)
2. **Train SAE**: Overcomplete autoencoder (768 → 6144 → 768) with L1 sparsity penalty. Architecture follows [Anthropic's "Towards Monosemanticity"](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
3. **Analyze features**: Find maximally activating examples for each feature, compare interpretability with SVD components

## References

- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) — Anthropic, 2023
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) — Anthropic, 2024
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — Anthropic, 2022

## Author

Tim van Driel, Ph.D. — Lead Scientist, SLAC National Accelerator Laboratory
