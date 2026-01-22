# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tsilva-notebook-utils is a utility library for Jupyter notebooks and Google Colab, providing tools for video rendering, RL utilities, PyTorch Lightning callbacks, visualization, and integration with external services (HuggingFace, W&B, OpenRouter).

## Commands

```bash
# Testing
make test              # Full test suite with linting (black, isort, flake8) and coverage
make test-fast         # Tests only, no linting
make test-unit         # Unit tests only
pytest tests/test_gymnasium.py::TestSyncRolloutCollector -v  # Run specific test class

# Code quality
make lint              # Check formatting (black, isort, flake8)
make format            # Auto-format code

# Version management
make bump-patch        # 0.0.x → 0.0.x+1
make bump-minor        # 0.x.y → 0.x+1.0
make bump-major        # x.y.z → x+1.0.0

# Build and publish
make build             # Build package
make publish           # Build and publish to PyPI
```

## Architecture

**Lazy imports pattern**: Modules are not imported by default in `__init__.py` to avoid heavy dependencies. Import modules directly: `from tsilva_notebook_utils.gymnasium import ...`

**Optional dependencies**: Each module handles missing dependencies gracefully with try/except fallbacks. Feature groups are defined in `pyproject.toml` (e.g., `pip install -e .[gymnasium]`).

**Key modules by domain**:
- `gymnasium.py` - RL utilities: episode recording, rollout collectors (sync/async), Lightning callbacks for RL agents
- `lightning.py` - PyTorch Lightning: callbacks (ThresholdStopping, BackboneWarmup, EpochTimeLogger), dataset specs, transforms
- `plots.py` - Visualization: matplotlib/seaborn/bokeh plots, UMAP embeddings, confusion matrices
- `torch.py` - PyTorch: device detection, GPU stats, weight initialization
- `video.py` - Video rendering from frames with labeling support
- `colab.py` - Google Colab: auto-disconnect, secrets, notebook ID extraction

## Code Standards

- **Black**: 88 char line length, Python 3.9+ target
- **isort**: black profile
- **flake8**: max complexity 10, line length 127
- **Coverage**: 80% target

Version is tracked in both `pyproject.toml` and `tsilva_notebook_utils/__init__.py` via bump2version.

## Important Notes

- README.md must be kept up to date with any significant project changes
