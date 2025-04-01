# tsilva-notebook-utils

Utility functions for Jupyter notebooks.

## Installation

```bash
git clone https://github.com/tsilva/tsilva-notebook-utils.git
cd tsilva-notebook-utils
curl -L https://gist.githubusercontent.com/tsilva/258374c1ba2296d8ba22fffbf640f183/raw/venv-install.sh -o install.sh && chmod +x install.sh && ./install.sh
```

## Usage

```python
from tsilva_notebook_utils import render_video

# Use the functions as needed
```

## Publish

```
pip install build twine
python -m build
python -m twine upload dist/*
```

## Publishing

This package uses a script to automate the publishing process to PyPI. The script handles version bumping, building, and uploading the package.

### Prerequisites

Ensure you have the necessary tools installed:

```bash
pip install twine build wheel
```

Also make sure you have configured PyPI credentials in your `~/.pypirc` file or through environment variables.

### Publishing a New Release

To publish a new version:

1. Make sure all your changes are committed to git
2. Run the publish script:

```bash
# For a patch release (0.1.7 -> 0.1.8)
python publish.py

# For a minor release (0.1.7 -> 0.2.0)
python publish.py --bump minor

# For a major release (0.1.7 -> 1.0.0)
python publish.py --bump major
```

3. Push the changes and tags to GitHub:

```bash
git push && git push --tags
```

### Options

The publish script supports the following options:

- `--bump {patch,minor,major}`: Specifies the version increment type (default: patch)
- `--no-upload`: Builds the package but skips uploading to PyPI

### Example

```bash
# Build a minor release but don't publish yet
python publish.py --bump minor --no-upload
```

The script will:
1. Check for uncommitted changes
2. Bump the version in setup.py
3. Build the package (sdist and wheel)
4. Upload to PyPI (unless --no-upload is specified)
5. Commit the version change and create a git tag