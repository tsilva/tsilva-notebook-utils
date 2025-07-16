.PHONY: help install test bump-patch bump-minor bump-major build publish clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e .[dev]

test:  ## Run tests with coverage and linting
	./run_tests.sh

test-fast:  ## Run tests only (no linting)
	python -m pytest tests/ --cov=tsilva_notebook_utils --cov-report=term-missing -v

test-unit:  ## Run unit tests only
	python -m pytest tests/ -v

lint:  ## Run linting checks only
	black --check --diff .
	isort --check-only --diff .
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

format:  ## Format code with black and isort
	black .
	isort .

bump-patch:  ## Bump patch version (0.0.1 -> 0.0.2)
	python bump_version.py patch

bump-minor:  ## Bump minor version (0.1.0 -> 0.2.0)
	python bump_version.py minor

bump-major:  ## Bump major version (1.0.0 -> 2.0.0)
	python bump_version.py major

build:  ## Build the package
	python -m build

publish: build  ## Build and publish to PyPI
	python -m twine upload dist/*

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
