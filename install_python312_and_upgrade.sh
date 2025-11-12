#!/bin/bash
# install_python312_and_upgrade.sh - Install Python 3.12 and upgrade environment

set -e  # Exit on error

echo "🚀 Python 3.12 Installation & Environment Upgrade"
echo "=================================================="

# Detect architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Step 1: Download Python 3.12
echo ""
echo "Step 1: Downloading Python 3.12..."

if [ "$ARCH" = "arm64" ]; then
    # Apple Silicon (M1/M2/M3)
    PYTHON_URL="https://www.python.org/ftp/python/3.12.8/python-3.12.8-macos11.pkg"
    echo "Detected Apple Silicon - downloading ARM64 version"
else
    # Intel
    PYTHON_URL="https://www.python.org/ftp/python/3.12.8/python-3.12.8-macosx10.9.pkg"
    echo "Detected Intel - downloading x86_64 version"
fi

PYTHON_PKG="/tmp/python-3.12.8.pkg"

if [ -f "$PYTHON_PKG" ]; then
    echo "✅ Python installer already downloaded"
else
    echo "Downloading from $PYTHON_URL..."
    curl -L -o "$PYTHON_PKG" "$PYTHON_URL"
    echo "✅ Downloaded to $PYTHON_PKG"
fi

# Step 2: Install Python 3.12
echo ""
echo "Step 2: Installing Python 3.12..."
echo "⚠️  This requires sudo privileges. You may be prompted for your password."
sudo installer -pkg "$PYTHON_PKG" -target /

echo "✅ Python 3.12 installed"

# Verify installation
echo ""
echo "Verifying Python 3.12 installation..."
if command -v python3.12 &> /dev/null; then
    python3.12 --version
else
    echo "❌ Python 3.12 not found in PATH"
    echo "Trying /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12..."
    if [ -f "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12" ]; then
        PYTHON312="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
        $PYTHON312 --version
        echo "Adding to PATH for this session..."
        export PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:$PATH"
    else
        echo "❌ Python 3.12 installation failed"
        exit 1
    fi
fi

# Step 3: Backup current venv
echo ""
echo "Step 3: Backing up current virtual environment..."
cd "$(dirname "$0")"
if [ -d ".venv" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    mv .venv .venv_backup_$timestamp
    echo "✅ Backed up to .venv_backup_$timestamp"
else
    echo "⚠️  No existing .venv found"
fi

# Step 4: Create new venv with Python 3.12
echo ""
echo "Step 4: Creating new virtual environment with Python 3.12..."
python3.12 -m venv .venv
echo "✅ Virtual environment created"

# Step 5: Activate and upgrade pip
echo ""
echo "Step 5: Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip
echo "✅ pip upgraded"

# Step 6: Install latest anomalib
echo ""
echo "Step 6: Installing latest Anomalib (2.x) with Lightning 2.x support..."
.venv/bin/pip install "anomalib>=2.0.0"
echo "✅ Anomalib installed"

# Step 7: Install other core dependencies
echo ""
echo "Step 7: Installing core dependencies..."
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

echo "✅ Core dependencies installed"

# Step 8: Verify installation
echo ""
echo "Step 8: Verifying installation..."
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
echo "Summary:"
echo "  ✅ Python 3.12 installed"
echo "  ✅ New .venv created with Python 3.12"
echo "  ✅ Anomalib 2.x installed (with Lightning 2.x)"
echo "  ✅ Old environment backed up"
echo ""
echo "Next steps:"
echo "  1. Activate: source .venv/bin/activate"
echo "  2. Update train.py for Anomalib 2.x API"
echo "  3. Test: .venv/bin/python train.py"
echo ""
