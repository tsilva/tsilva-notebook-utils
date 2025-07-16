#!/bin/bash

# Script to run tests locally
echo "🧪 Running tsilva-notebook-utils tests..."

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest not found. Please install test dependencies:${NC}"
    echo "pip install -e .[test]"
    exit 1
fi

# Run linting checks
echo -e "${YELLOW}🔍 Running linting checks...${NC}"

# Check if black is available
if command -v black &> /dev/null; then
    echo "Checking code formatting with black..."
    black --check --diff .
    BLACK_EXIT=$?
else
    echo "⚠️  black not installed, skipping format check"
    BLACK_EXIT=0
fi

# Check if isort is available
if command -v isort &> /dev/null; then
    echo "Checking import sorting with isort..."
    isort --check-only --diff .
    ISORT_EXIT=$?
else
    echo "⚠️  isort not installed, skipping import sort check"
    ISORT_EXIT=0
fi

# Check if flake8 is available
if command -v flake8 &> /dev/null; then
    echo "Running flake8 linting..."
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    FLAKE8_EXIT=$?
else
    echo "⚠️  flake8 not installed, skipping lint check"
    FLAKE8_EXIT=0
fi

# Run tests
echo -e "${YELLOW}🧪 Running tests...${NC}"
pytest tests/ --cov=tsilva_notebook_utils --cov-report=term-missing --cov-report=html -v
PYTEST_EXIT=$?

# Summary
echo ""
echo "📊 Test Results Summary:"
echo "========================"

if [ $BLACK_EXIT -eq 0 ]; then
    echo -e "✅ Code formatting: ${GREEN}PASSED${NC}"
else
    echo -e "❌ Code formatting: ${RED}FAILED${NC}"
fi

if [ $ISORT_EXIT -eq 0 ]; then
    echo -e "✅ Import sorting: ${GREEN}PASSED${NC}"
else
    echo -e "❌ Import sorting: ${RED}FAILED${NC}"
fi

if [ $FLAKE8_EXIT -eq 0 ]; then
    echo -e "✅ Linting: ${GREEN}PASSED${NC}"
else
    echo -e "❌ Linting: ${RED}FAILED${NC}"
fi

if [ $PYTEST_EXIT -eq 0 ]; then
    echo -e "✅ Unit tests: ${GREEN}PASSED${NC}"
else
    echo -e "❌ Unit tests: ${RED}FAILED${NC}"
fi

# Overall result
OVERALL_EXIT=$((BLACK_EXIT + ISORT_EXIT + FLAKE8_EXIT + PYTEST_EXIT))

if [ $OVERALL_EXIT -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    echo "Coverage report available in htmlcov/index.html"
else
    echo ""
    echo -e "${RED}💥 Some tests failed. Please check the output above.${NC}"
fi

exit $OVERALL_EXIT
