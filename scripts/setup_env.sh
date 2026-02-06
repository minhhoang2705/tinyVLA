#!/bin/bash
# Environment setup script for tinyVLA
# Creates virtual environment with PyTorch + CUDA and installs dependencies

set -e  # Exit on error

echo "==================================="
echo "tinyVLA Environment Setup"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Project root: $PROJECT_ROOT${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo -e "${YELLOW}Detected Python version: $PYTHON_VERSION${NC}"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
uv pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
echo -e "${YELLOW}Installing PyTorch 2.5.0 with CUDA 11.8...${NC}"
uv pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# Install project with development dependencies
echo -e "${YELLOW}Installing tinyVLA with dev dependencies...${NC}"
uv pip install -e ".[dev]"

echo ""
echo -e "${GREEN}==================================="
echo "Installation Complete!"
echo "===================================${NC}"
echo ""
echo "To verify installation, run:"
echo "  source .venv/bin/activate"
echo "  python scripts/verify_install.py"
echo ""
echo "To activate this environment in the future:"
echo "  source .venv/bin/activate"
echo ""
