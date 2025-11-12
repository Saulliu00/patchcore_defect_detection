#!/bin/bash
# upgrade_to_python314.sh - Upgrade environment to Python 3.14 with latest Anomalib

set -e  # Exit on error

echo "🚀 Upgrading to Python 3.14 with Anomalib 2.x"
echo "=============================================="

cd "$(dirname "$0")"

# Verify Python 3.14
echo ""
echo "Step 1: Verifying Python 3.14..."
if ! command -v python3.14 &> /dev/null; then
    echo "❌ Python 3.14 not found"
    exit 1
fi
python3.14 --version
echo "✅ Python 3.14 found"

# Backup current venv
echo ""
echo "Step 2: Backing up current virtual environment..."
if [ -d ".venv" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    mv .venv .venv_backup_$timestamp
    echo "✅ Backed up to .venv_backup_$timestamp"
else
    echo "⚠️  No existing .venv found"
fi

# Create new venv with Python 3.14
echo ""
echo "Step 3: Creating new virtual environment with Python 3.14..."
python3.14 -m venv .venv
echo "✅ Virtual environment created"

# Upgrade pip
echo ""
echo "Step 4: Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip
echo "✅ pip upgraded"

# Install latest anomalib
echo ""
echo "Step 5: Installing latest Anomalib (2.x)..."
.venv/bin/pip install "anomalib>=2.0.0"
echo "✅ Anomalib installed"

# Install other dependencies
echo ""
echo "Step 6: Installing additional dependencies..."
.venv/bin/pip install \
    torch \
    torchvision \
    opencv-python \
    pillow \
    matplotlib \
    scikit-learn \
    pyyaml \
    tqdm \
    tensorboard

echo "✅ Dependencies installed"

# Verify installation
echo ""
echo "Step 7: Verifying installation..."
echo "=================================="
.venv/bin/python << 'PYEOF'
import sys
print(f"Python version: {sys.version}")
import anomalib
print(f"Anomalib version: {anomalib.__version__}")
import lightning
print(f"Lightning version: {lightning.__version__}")
import torch
print(f"PyTorch version: {torch.__version__}")
PYEOF

echo ""
echo "✅ Verification complete"

echo ""
echo "🎉 Environment upgrade successful!"
echo ""
echo "Summary:"
echo "  ✅ Python 3.14 environment created"
echo "  ✅ Anomalib 2.x installed"
echo "  ✅ Lightning 2.x installed"
echo "  ✅ Old environment backed up"
echo ""
echo "Next steps:"
echo "  1. I will now update train.py for Anomalib 2.x API"
echo ""
