# ESRecsys — Efficient & Scalable Recommendation Systems

A companion codebase for the book **Building Recommendation Systems in Python and JAX** (Bryan Bischoff & Hector Yee).

This is a **fork** of the [original repository](https://github.com/BBischof/ESRecsys), migrated from TensorFlow/JAX/Flax to **PyTorch** with the latest framework versions and Python 3.11+.

## Overview

| Sub-project | Model | ML Task | Framework |
|---|---|---|---|
| **pinterest/** | Two-tower CNN | Scene-based complementary product recommendation | PyTorch |
| **spotify/** | Embedding retrieval | Playlist continuation (next-track prediction) | PyTorch |
| **wikipedia/** | GloVe + LSTM | Word/URL embedding & text-to-URL retrieval | PyTorch + PySpark |

## Quick Start

```bash
conda create -n esrecsys python=3.11 -y
conda activate esrecsys
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install wandb absl-py numpy
```

See each sub-project's README for data preparation and training instructions.

## Directory Layout

```
├── pinterest/          # Shop The Look — visual complementary product recommender
├── spotify/            # Million Playlist Dataset — playlist continuation
├── wikipedia/          # Wikipedia NLP pipeline — word & URL embeddings
├── proto/              # Shared protobuf definitions
├── book-text/          # LaTeX book chapters (no code)
```

## Requirements

- Python 3.11+
- PyTorch 2.x (CUDA recommended for pinterest/wikipedia training)
- Java Runtime (for PySpark data processing in wikipedia)
- 16GB+ RAM (for wikipedia PySpark pipeline)

## License

Apache 2.0 — see individual file headers.
