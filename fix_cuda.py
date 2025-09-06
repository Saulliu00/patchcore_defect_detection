import subprocess
import sys

def fix_pytorch_cuda():
    print("🔧 Fixing PyTorch CUDA installation...")
    
    # Uninstall current PyTorch
    print("Removing current PyTorch...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", 
                   "torch", "torchvision", "torchaudio", "-y"])
    
    # Install CUDA version
    print("Installing PyTorch with CUDA 11.8...")
    subprocess.run([sys.executable, "-m", "pip", "install",
                   "torch", "torchvision", "torchaudio",
                   "--index-url", "https://download.pytorch.org/whl/cu118"])
    
    # Verify
    print("Verifying installation...")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    fix_pytorch_cuda()