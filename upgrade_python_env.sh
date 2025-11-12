#!/bin/bash
# upgrade_python_env.sh - Upgrade to Python 3.13 and latest Anomalib

set -e  # Exit on error

echo "🚀 Python Environment Upgrade Script"
echo "======================================"

# Step 1: Check for Python 3.13
echo ""
echo "Step 1: Checking for Python 3.13..."
if command -v python3.13 &> /dev/null; then
    echo "✅ Python 3.13 found: $(python3.13 --version)"
else
    echo "❌ Python 3.13 not found"
    echo ""
    echo "Please install Python 3.13 first:"
    echo ""
    echo "Option 1 - Homebrew (Recommended):"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "  brew install python@3.13"
    echo ""
    echo "Option 2 - Download from python.org:"
    echo "  Visit: https://www.python.org/downloads/"
    echo ""
    exit 1
fi

# Step 2: Backup current venv
echo ""
echo "Step 2: Backing up current virtual environment..."
if [ -d ".venv" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    mv .venv .venv_backup_$timestamp
    echo "✅ Backed up to .venv_backup_$timestamp"
else
    echo "⚠️  No existing .venv found"
fi

# Step 3: Create new venv with Python 3.13
echo ""
echo "Step 3: Creating new virtual environment with Python 3.13..."
python3.13 -m venv .venv
echo "✅ Virtual environment created"

# Step 4: Activate and upgrade pip
echo ""
echo "Step 4: Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip
echo "✅ pip upgraded"

# Step 5: Install latest anomalib
echo ""
echo "Step 5: Installing latest Anomalib (2.x)..."
.venv/bin/pip install "anomalib>=2.0.0"
echo "✅ Anomalib installed"

# Step 6: Install other core dependencies
echo ""
echo "Step 6: Installing core dependencies..."
.venv/bin/pip install \
    torch \
    torchvision \
    opencv-python \
    pillow \
    matplotlib \
    scikit-learn \
    pyyaml \
    tqdm

echo "✅ Core dependencies installed"

# Step 7: Verify installation
echo ""
echo "Step 7: Verifying installation..."
echo "=================================="
.venv/bin/python -c "
import sys
print(f'Python version: {sys.version}')
import anomalib
print(f'Anomalib version: {anomalib.__version__}')
import lightning
print(f'Lightning version: {lightning.__version__}')
import torch
print(f'PyTorch version: {torch.__version__}')
"
echo "✅ Verification complete"

echo ""
echo "🎉 Environment upgrade complete!"
echo ""
echo "Next steps:"
echo "1. Your old environment is backed up"
echo "2. Activate the new environment: source .venv/bin/activate"
echo "3. Update train.py to use the new Anomalib 2.x API"
echo ""
