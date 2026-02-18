> [!WARNING]
> ## Archived
> This project is archived and no longer maintained.
>
> Development has shifted to local Jupyter notebook workflows with in-repo utility imports and remote kernels when needed. No further updates or fixes are planned.

<div align="center">
  <img src="logo.png" alt="tsilva-notebook-utils" width="512"/>

  [![Tests](https://github.com/tsilva/tsilva-notebook-utils/actions/workflows/tests.yml/badge.svg)](https://github.com/tsilva/tsilva-notebook-utils/actions/workflows/tests.yml)
  [![codecov](https://codecov.io/gh/tsilva/tsilva-notebook-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/tsilva/tsilva-notebook-utils)
  [![PyPI version](https://badge.fury.io/py/tsilva-notebook-utils.svg)](https://badge.fury.io/py/tsilva-notebook-utils)
  [![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/tsilva/tsilva-notebook-utils)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

  **📓 A comprehensive toolkit for Jupyter notebooks and Google Colab with utilities for RL, PyTorch Lightning, visualization, and video rendering 🚀**

</div>

## Overview

tsilva-notebook-utils provides a modular collection of utilities designed to streamline machine learning workflows in Jupyter notebooks and Google Colab environments. Each module handles missing dependencies gracefully, allowing you to install only what you need.

## Features

- **Reinforcement Learning** — Episode recording, rollout collectors (sync/async), GAE computation, and PyTorch Lightning callbacks for RL training
- **PyTorch Lightning** — Callbacks for threshold-based stopping, backbone warmup, epoch timing, and W&B cleanup
- **Data Modules** — Ready-to-use MNIST and CIFAR10 data modules with configurable augmentation pipelines
- **Visualization** — Interactive embedding plots (t-SNE, UMAP, PCA) with Bokeh, heatmaps, and matplotlib utilities
- **Video Rendering** — Create videos from frame sequences with labels, grid layouts, and model prediction comparisons
- **PyTorch Utilities** — Device detection (CUDA/MPS/CPU), GPU stats, weight initialization, gradient norms, and filter visualization
- **Google Colab** — Auto-disconnect after idle timeout, secrets management, and notebook ID extraction
- **Integrations** — HuggingFace Hub, Weights & Biases, and OpenRouter API support

## 🚀 Quick Start

```bash
pip install tsilva-notebook-utils
```

```python
# Import modules directly (lazy imports to avoid heavy dependencies)
from tsilva_notebook_utils.gymnasium import render_episode, collect_rollouts
from tsilva_notebook_utils.lightning import ThresholdStoppingCallback
from tsilva_notebook_utils.video import create_video_from_frames
```

## 📦 Installation

Install the base package or with specific feature groups:

```bash
# Base installation
pip install tsilva-notebook-utils

# With specific features
pip install tsilva-notebook-utils[gymnasium]   # RL utilities
pip install tsilva-notebook-utils[lightning]   # PyTorch Lightning
pip install tsilva-notebook-utils[plots]       # Visualization
pip install tsilva-notebook-utils[video]       # Video rendering
pip install tsilva-notebook-utils[colab]       # Google Colab utilities

# All features
pip install tsilva-notebook-utils[all]
```

## 📖 Usage

### Reinforcement Learning

```python
from tsilva_notebook_utils.gymnasium import (
    build_env,
    render_episode,
    collect_rollouts,
    SyncRolloutCollector
)

# Build vectorized environment
env = build_env("CartPole-v1", n_envs=4, seed=42)

# Render an episode as embedded video
video = render_episode(env, model, seed=42, fps=30)
display(video)

# Collect rollouts with GAE
trajectories, extras = collect_rollouts(
    env, policy_model, value_model,
    n_steps=1024, gamma=0.99, lam=0.95
)
```

### PyTorch Lightning Callbacks

```python
from tsilva_notebook_utils.lightning import (
    ThresholdStoppingCallback,
    BackboneWarmupCallback,
    EpochTimeLogger,
    WandbCleanup
)

callbacks = [
    ThresholdStoppingCallback(metric="val/acc", threshold=0.95),
    BackboneWarmupCallback(unfreeze_at=0.3),  # Unfreeze at 30% of training
    EpochTimeLogger(),
    WandbCleanup()  # Clean shutdown of W&B in notebooks
]

trainer = pl.Trainer(callbacks=callbacks)
```

### Data Modules

```python
from tsilva_notebook_utils.lightning import (
    CIFAR10DataModule,
    create_data_module,
    render_samples_per_class
)

# Create data module with augmentation
dm = create_data_module({
    "dataset_id": "cifar10",
    "seed": 42,
    "batch_size": 64,
    "train_size": 0.9,
    "augmentation_pipeline": [
        ("RandomHorizontalFlip", [], {"p": 0.5}),
        ("RandomRotation", [15], {})
    ]
})

# Visualize samples per class
render_samples_per_class(dm, n_samples=5, split='train')
```

### Visualization

```python
from tsilva_notebook_utils.plots import (
    plot_embeddings_with_inputs,
    plot_vector_batch_heatmap,
    plot_series
)

# Interactive embedding visualization with hover previews
plot_embeddings_with_inputs(
    embeddings, raw_images,
    embedding_method="umap",
    captions=labels
)

# Heatmap of latent vectors
plot_vector_batch_heatmap(latent_vectors, title="Latent Space")
```

### Video Rendering

```python
from tsilva_notebook_utils.video import (
    create_video_from_frames,
    render_video_from_dataloader
)

# Create video from frames directory
video = create_video_from_frames("./frames", fps=30, scale=2)

# Render model predictions vs targets
video = render_video_from_dataloader(
    loader, model=model,
    x_key="x", y_key="y",
    fps=5, scale=4
)
```

### PyTorch Utilities

```python
from tsilva_notebook_utils.torch import (
    get_default_device,
    get_gpu_stats,
    apply_weight_init,
    get_model_parameter_counts,
    seed_everything
)

# Automatic device selection (CUDA > MPS > CPU)
device = get_default_device()

# GPU memory stats
stats = get_gpu_stats()
print(f"Free GPU memory: {stats['free_memory_gb']:.2f} GB")

# Weight initialization
model = apply_weight_init(model, weight_init='kaiming', nonlinearity='relu')

# Parameter counts
counts = get_model_parameter_counts(model)
print(f"Trainable: {counts['trainable']:,}")
```

### Google Colab

```python
from tsilva_notebook_utils.colab import (
    disconnect_after_timeout,
    load_secrets_into_env,
    notebook_id_from_title
)

# Auto-disconnect after 5 minutes of inactivity
disconnect_after_timeout(timeout_seconds=300)

# Load secrets from Colab userdata into environment
load_secrets_into_env(["API_KEY", "HF_TOKEN"])
```

## 🛠️ Development

```bash
# Clone and install in development mode
git clone https://github.com/tsilva/tsilva-notebook-utils.git
cd tsilva-notebook-utils
pip install -e .[dev]

# Run tests
make test          # Full suite with linting and coverage
make test-fast     # Tests only
make test-unit     # Unit tests only

# Code quality
make lint          # Check formatting
make format        # Auto-format code

# Version management
make bump-patch    # 0.0.x -> 0.0.x+1
make bump-minor    # 0.x.y -> 0.x+1.0
make bump-major    # x.y.z -> x+1.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`make test`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).
