# 🧰 tsilva-notebook-utils

[![Tests](https://github.com/tsilva/tsilva-notebook-utils/actions/workflows/tests.yml/badge.svg)](https://github.com/tsilva/tsilva-notebook-utils/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/tsilva/tsilva-notebook-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/tsilva/tsilva-notebook-utils)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/tsilva/tsilva-notebook-utils)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/tsilva-notebook-utils.svg)](https://badge.fury.io/py/tsilva-notebook-utils)

🔬 Handy utilities for enhancing your Jupyter and Google Colab notebooks

## 📖 Overview

`tsilva-notebook-utils` is a collection of utility functions designed to make working with Jupyter and Google Colab notebooks more efficient. It provides tools for video rendering, notification systems, and Colab-specific features like automatic disconnection after idle periods.

## 🛠️ Usage

### Video Rendering

```python
from tsilva_notebook_utils import render_video

# Render a simple video from frames
frames = [frame1, frame2, frame3]  # List of numpy arrays
video = render_video(frames, fps=30, scale=1.5)
display(video)

# Render frames with labels
labeled_frames = [(frame1, "Start"), (frame2, "Middle"), (frame3, "End")]
video = render_video(labeled_frames, fps=24)
display(video)

# Compare multiple videos side by side
from tsilva_notebook_utils import render_videos
render_videos([(video1_frames, "Original"), (video2_frames, "Processed")])
```

### Google Colab Utilities

```python
from tsilva_notebook_utils import disconnect_after_timeout

# Automatically disconnect Colab after 5 minutes of inactivity
disconnect_after_timeout(timeout_seconds=300)
```

### Notifications

Send notifications to [PopDesk](https://github.com/tsilva/popdesk) notification server:

```python
from tsilva_notebook_utils import send_popdesk_notification

# Send a notification when your long-running notebook task completes
send_popdesk_notification(
    url="https://your-popdesk-url",
    auth_token="your-auth-token",
    title="Training Complete",
    message="Your model has finished training with 95% accuracy"
)
```

## � Development & Releases

### Development Setup

```bash
# Clone the repository
git clone https://github.com/tsilva/tsilva-notebook-utils.git
cd tsilva-notebook-utils

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

### Version Management

This project uses automated version bumping with `bump2version`. Version numbers are automatically updated across all relevant files.

#### Local Version Bumping

Use the provided scripts for local development:

```bash
# Bump patch version (0.0.115 → 0.0.118)
make bump-patch
# or
python bump_version.py patch

# Bump minor version (0.0.115 → 0.1.0)
make bump-minor
# or
python bump_version.py minor

# Bump major version (0.0.115 → 1.0.0)
make bump-major
# or
python bump_version.py major
```

#### Automated Releases

**🎯 Recommended: GitHub Actions Release Workflow**

1. Go to the [GitHub Actions](../../actions) tab
2. Select "Release and Publish" workflow
3. Click "Run workflow"
4. Choose your release type:
   - **patch**: Bug fixes (0.0.115 → 0.0.118)
   - **minor**: New features (0.0.115 → 0.1.0)
   - **major**: Breaking changes (0.0.115 → 1.0.0)
   - **prerelease**: Beta versions (0.0.115 → 0.0.118-alpha.1)
5. Click "Run workflow"

**What happens automatically:**
- ✅ Version bumped in `pyproject.toml` and `__init__.py`
- ✅ Git commit created with version bump message
- ✅ Git tag created (e.g., `v0.0.118`)
- ✅ Package built and published to [PyPI](https://pypi.org/project/tsilva-notebook-utils/)
- ✅ GitHub release created with release notes

#### Manual Release Process

If you prefer manual control:

```bash
# 1. Bump version locally
make bump-patch  # or bump-minor, bump-major

# 2. Build the package
make build

# 3. Publish to PyPI (requires PYPI_API_TOKEN)
make publish
```

### Release Notes

Recent releases can be found on the [Releases page](../../releases).

#### Version History
- **v0.0.115**: Current version with automated release workflow
- **v0.0.114**: Previous stable version

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`./run_tests.sh` or `make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 🧪 Testing

### Running Tests Locally

```bash
# Install test dependencies
pip install -e .[test]

# Run all tests with coverage
./run_tests.sh

# Or use pytest directly
pytest tests/ --cov=tsilva_notebook_utils --cov-report=term-missing -v

# Run specific test file
pytest tests/test_gymnasium.py -v

# Run tests for specific class
pytest tests/test_gymnasium.py::TestSyncRolloutCollector -v
```

### Test Structure

The test suite includes:
- **Unit tests** for individual functions and classes
- **Integration tests** for complex workflows
- **Coverage reporting** to ensure code quality
- **Linting checks** with flake8, black, and isort

Tests are automatically run on:
- Every push to `main` and `develop` branches
- Every pull request
- Multiple Python versions (3.9, 3.10, 3.11, 3.12)

### Available Make Commands

```bash
make help          # Show all available commands
make install       # Install package in development mode
make test          # Run tests
make bump-patch    # Bump patch version
make bump-minor    # Bump minor version
make bump-major    # Bump major version
make build         # Build the package
make publish       # Build and publish to PyPI
make clean         # Clean build artifacts
```

## �📄 License

This project is licensed under the [MIT License](LICENSE).