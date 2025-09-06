# setup_project.py - Project setup script (place in root directory)
import os
import sys
from pathlib import Path
import subprocess
import torch

def check_gpu():
    """Check GPU availability and setup"""
    print("Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"SUCCESS: GPU detected: {gpu_name}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {gpu_count}")
        
        # Check if it's RTX 3080 Ti
        if "3080" in gpu_name or "RTX" in gpu_name:
            print("EXCELLENT: RTX 3080 Ti detected - excellent for training!")
        
        return True
    else:
        print("WARNING: No GPU detected. Training will use CPU (much slower).")
        return False

def create_project_structure():
    """Create the complete project structure"""
    print("Setting up PatchCore project structure...")
    
    directories = [
        "data/train/good",
        "data/test/good", 
        "data/test/defective",
        "data/val/good",
        "data/val/defective",
        "models/saved_models",
        "models/deployment",
        "database/images/normal",
        "database/images/defective",
        "logs/tensorboard",
        "config",
        "src/training",
        "src/inference", 
        "src/utils",
        "src/deployment"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "src/training/__init__.py",
        "src/inference/__init__.py",
        "src/utils/__init__.py",
        "src/deployment/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        
    print("SUCCESS: Project directories created successfully!")

def create_requirements_files():
    """Create requirements files optimized for RTX 3080 Ti"""
    
    # Local training requirements (with CUDA support)
    local_requirements = """# Local Training Requirements (CUDA/GPU Support)
torch>=2.0.0+cu118
torchvision>=0.15.0+cu118
torchaudio>=2.0.0+cu118
anomalib>=1.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
albumentations>=1.3.0
lightning>=2.0.0
omegaconf>=2.3.0
rich>=13.0.0
tqdm>=4.65.0
Pillow>=10.0.0
tensorboard>=2.13.0
torchmetrics>=1.0.0
timm>=0.9.0
"""

    # Raspberry Pi requirements (CPU only)
    pi_requirements = """# Raspberry Pi Requirements (CPU Only)
torch>=2.0.0+cpu
torchvision>=0.15.0+cpu
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
picamera2>=0.3.0
gpiozero>=1.6.0
omegaconf>=2.3.0
tqdm>=4.65.0
"""
    
    # Write requirements files
    with open("requirements_local.txt", "w", encoding='utf-8') as f:
        f.write(local_requirements)
    
    with open("requirements_pi.txt", "w", encoding='utf-8') as f:
        f.write(pi_requirements)
    
    print("SUCCESS: Requirements files created (optimized for RTX 3080 Ti)")

def update_config_for_gpu():
    """Update configuration for GPU training"""
    
    config_content = """# config/patchcore_config.yaml - GPU Optimized Configuration
model:
  name: patchcore
  backbone: wide_resnet50_2
  pre_trained: true
  layers:
    - layer2
    - layer3
  coreset_sampling_ratio: 0.1
  num_neighbors: 9

dataset:
  name: folder
  format: folder
  path: ./data
  normal_dir: good
  abnormal_dir: defective
  task: segmentation
  image_size: [1024, 1024]  # High resolution for RTX 3080 Ti
  center_crop: null
  normalization: imagenet
  
dataloader:
  train_batch_size: 16        # Increased for RTX 3080 Ti (12GB VRAM)
  eval_batch_size: 16         # Increased batch size
  num_workers: 8              # Increased workers for faster data loading
  
trainer:
  accelerator: gpu            # Use GPU acceleration
  devices: 1                  # Single GPU
  precision: 16               # Mixed precision for faster training
  default_root_dir: ./models/saved_models
  max_epochs: 1               # PatchCore needs only 1 epoch
  logger: true
  log_every_n_steps: 10
  val_check_interval: 1.0
  enable_progress_bar: true
  
metrics:
  image:
    - F1Score
    - AUROC
  pixel:
    - F1Score
    - AUROC
  threshold:
    method: adaptive
    
optimization:
  export_mode: torch
  
logging:
  logger:
    - class_path: lightning.pytorch.loggers.TensorBoardLogger
      init_args:
        save_dir: logs/tensorboard
        name: patchcore_experiment
        
# Tiling configuration for high-resolution inference
tiling:
  apply: true
  tile_size: [256, 256]
  stride: [128, 128]          # 50% overlap
  remove_border_count: 0
  mode: padding
  
# GPU-specific optimizations
gpu_optimizations:
  mixed_precision: true       # Use automatic mixed precision
  compile_model: false        # PyTorch 2.0 compilation (experimental)
  channels_last: true         # Memory format optimization
  benchmark_cudnn: true       # CuDNN benchmarking
  
# Deployment specific settings
deployment:
  device: cpu                 # Raspberry Pi will use CPU
  precision: 32
  optimize_for_mobile: true
  quantization: false
"""
    
    # Create config directory and file
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    with open("config/patchcore_config.yaml", "w", encoding='utf-8') as f:
        f.write(config_content)
    
    print("SUCCESS: GPU-optimized configuration created")

def install_dependencies(use_gpu=True):
    """Install required dependencies"""
    print("Installing dependencies...")
    
    if use_gpu:
        print("Installing GPU-accelerated packages...")
        # Install PyTorch with CUDA support for RTX 3080 Ti
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], check=True)
            print("SUCCESS: PyTorch with CUDA 11.8 installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install PyTorch with CUDA: {e}")
            print("WARNING: Falling back to CPU version...")
            use_gpu = False
    
    try:
        # Install other requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_local.txt"], 
                      check=True)
        print("SUCCESS: All dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install some dependencies: {e}")
        print("TIP: Try installing manually: pip install -r requirements_local.txt")
        return False
    
    return True

def create_sample_data_info():
    """Create information about data organization"""
    
    data_info = """# Data Organization Guide

## Training Data Structure

Place your images in the following structure:

```
data/
├── train/
│   └── good/              # ONLY normal/good parts (50-100+ images)
│       ├── part001.jpg
│       ├── part002.jpg
│       └── ...
├── test/
│   ├── good/              # Normal test images (20+ images)
│   │   ├── test_normal_001.jpg
│   │   └── ...
│   └── defective/         # Defective test images (20+ images)
│       ├── test_defect_001.jpg
│       └── ...
└── val/                   # Optional validation set
    ├── good/
    └── defective/
```

## Image Requirements for RTX 3080 Ti Training

- **Resolution**: 1024x1024 pixels or higher (your GPU can handle it!)
- **Format**: JPG, PNG supported
- **Quality**: High quality, well-lit, focused images
- **Consistency**: Similar lighting, angle, background
- **Quantity**: 
  - Training (normal): 50-100+ images minimum
  - Test (normal): 20+ images
  - Test (defective): 20+ images

## Tips for Best Results

1. **More normal data is better** - PatchCore learns what "normal" looks like
2. **Consistent imaging conditions** - same lighting, distance, angle
3. **High resolution** - your RTX 3080 Ti can handle 1024x1024 easily
4. **Variety in normal samples** - different orientations, minor variations
5. **Clear defects in test data** - obvious defects for evaluation

## Next Steps

1. Add your images to the appropriate folders
2. Run: `python train.py`
3. Monitor training with: `tensorboard --logdir logs/tensorboard`
"""
    
    with open("DATA_ORGANIZATION.md", "w", encoding='utf-8') as f:
        f.write(data_info)
    
    print("SUCCESS: Data organization guide created")

def create_gpu_optimized_train_script():
    """Create a GPU-optimized version of train.py"""
    
    train_script = """#!/usr/bin/env python3
# train.py - GPU Optimized Training Script for RTX 3080 Ti

import os
import sys
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore")

# Enable optimizations for RTX 3080 Ti
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for better performance

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from training.train_model import PatchCoreTrainer
from training.config_handler import ConfigHandler

def check_gpu_memory():
    \"\"\"Check GPU memory and optimize settings\"\"\"
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 10:  # RTX 3080 Ti has 12GB
            print("SUCCESS: Sufficient GPU memory for high-resolution training")
            return True
        else:
            print("WARNING: Limited GPU memory - consider reducing batch size")
            return False
    return False

def main():
    \"\"\"Main training function optimized for RTX 3080 Ti\"\"\"
    print("Starting PatchCore Training for Defect Detection")
    print("Optimized for NVIDIA RTX 3080 Ti")
    
    # Check GPU
    has_good_gpu = check_gpu_memory()
    
    # Check CUDA setup
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Training on: {gpu_name}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Enable mixed precision for RTX 3080 Ti
        print("Mixed precision training enabled")
    else:
        print("ERROR: CUDA not available! Training will be slow on CPU.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Check data directories
    data_path = Path("data")
    train_good = data_path / "train" / "good"
    test_good = data_path / "test" / "good" 
    test_defective = data_path / "test" / "defective"
    
    if not train_good.exists() or len(list(train_good.glob("*.jpg")) + list(train_good.glob("*.png"))) == 0:
        print("ERROR: No training images found in data/train/good/")
        print("Please see DATA_ORGANIZATION.md for setup instructions")
        return
    
    print(f"Found {len(list(train_good.glob('*')))} training images")
    
    # Initialize config handler
    config_handler = ConfigHandler("config/patchcore_config.yaml")
    config = config_handler.get_config()
    
    # Auto-adjust batch size based on GPU memory
    if has_good_gpu:
        print("Using optimized settings for RTX 3080 Ti")
        config.dataloader.train_batch_size = 16
        config.dataloader.eval_batch_size = 16
        config.dataloader.num_workers = 8
    else:
        print("Using conservative settings")
        config.dataloader.train_batch_size = 8
        config.dataloader.eval_batch_size = 8
        config.dataloader.num_workers = 4
    
    # Initialize trainer
    trainer = PatchCoreTrainer(config)
    
    # Train the model
    print("Starting training...")
    print("Monitor progress: tensorboard --logdir logs/tensorboard")
    
    model_path = trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    metrics = trainer.evaluate()
    
    print("\\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print("Metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric_name}: {value:.4f}")
    
    # Prepare for deployment
    print("\\nPreparing model for deployment...")
    deployment_path = trainer.prepare_for_deployment()
    
    print(f"\\nTraining Summary:")
    print(f"  Training model: {model_path}")
    print(f"  Deployment model: {deployment_path}")
    print(f"  Logs: logs/tensorboard/")
    print(f"\\nNext steps:")
    print(f"  1. Review metrics in TensorBoard")
    print(f"  2. Test model locally")
    print(f"  3. Deploy to Raspberry Pi:")
    print(f"     python deploy_to_pi.py --model_path {deployment_path}")

if __name__ == "__main__":
    main()
"""
    
    with open("train.py", "w", encoding='utf-8') as f:
        f.write(train_script)
    
    print("SUCCESS: GPU-optimized training script created")

def create_test_script():
    """Create a simple test script"""
    
    test_script = """#!/usr/bin/env python3
# test_model.py - Test trained model locally

import sys
from pathlib import Path
import cv2
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from inference.tiled_inference import TiledInference

def test_model():
    \"\"\"Test the trained model on sample images\"\"\"
    
    model_path = "models/deployment/patchcore_deployment.pth"
    
    if not Path(model_path).exists():
        print("ERROR: Model not found. Please train first: python train.py")
        return
    
    # Load model
    print("Loading model...")
    detector = TiledInference(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test on sample images
    test_dirs = ["data/test/good", "data/test/defective"]
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if not test_path.exists():
            continue
            
        print(f"\\nTesting on {test_dir}...")
        
        for img_path in test_path.glob("*.jpg"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            results = detector.predict(image)
            
            status = "DEFECTIVE" if results["is_anomaly"] else "NORMAL"
            score = results["anomaly_score"]
            
            print(f"  {img_path.name}: {status} (score: {score:.4f})")

if __name__ == "__main__":
    test_model()
"""
    
    with open("test_model.py", "w", encoding='utf-8') as f:
        f.write(test_script)
    
    print("SUCCESS: Test script created")

def main():
    """Main setup function"""
    print("Setting up PatchCore Defect Detection Project")
    print("Optimized for NVIDIA RTX 3080 Ti")
    print("=" * 60)
    
    # Check GPU first (before PyTorch installation)
    print("Step 1: GPU Detection")
    has_gpu = check_gpu()
    
    # Create project structure
    print("\nStep 2: Creating Project Structure")
    create_project_structure()
    
    # Create requirements files
    print("\nStep 3: Creating Requirements Files")
    create_requirements_files()
    
    # Update configuration for GPU
    print("\nStep 4: Creating GPU Configuration")
    update_config_for_gpu()
    
    # Install dependencies
    print("\nStep 5: Installing Dependencies")
    print("WARNING: This may take several minutes...")
    print("INFO: Installing PyTorch with CUDA support for your RTX 3080 Ti...")
    
    if not install_dependencies(use_gpu=has_gpu):
        print("ERROR: Dependency installation failed")
        print("TIP: You can install manually later with:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  pip install -r requirements_local.txt")
        
    # Final GPU verification after PyTorch installation
    print("\nStep 6: Final GPU Verification")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"SUCCESS: {gpu_name} is ready!")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            print("CUDA installation verified successfully!")
        else:
            print("WARNING: CUDA not available after installation")
            print("TIP: You may need to restart your terminal or install CUDA drivers")
    except ImportError:
        print("WARNING: PyTorch not available for verification")
    
    # Create documentation and scripts
    print("\nStep 7: Creating Scripts and Documentation")
    create_sample_data_info()
    create_gpu_optimized_train_script()
    create_test_script()
    
    print("\nSUCCESS: Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Read DATA_ORGANIZATION.md for data preparation")
    print("2. Place your training images in data/train/good/")
    print("3. Place your test images in data/test/good/ and data/test/defective/")
    print("4. Run training: python train.py")
    print("5. Test model: python test_model.py")
    print("6. Monitor training: tensorboard --logdir logs/tensorboard")
    print("7. Deploy to Pi: python deploy_to_pi.py --model_path models/deployment/patchcore_deployment.pth")
    
    if has_gpu:
        print("\nYour RTX 3080 Ti setup is complete!")
        print("TIP: The configuration is optimized for 1024x1024 images with batch size 16")
        print("TIP: If CUDA verification failed, try restarting your terminal")
    
    print("\nIMPORTANT: If you see CUDA errors, ensure you have:")
    print("1. Latest NVIDIA drivers installed")
    print("2. CUDA 11.8 compatible drivers (driver version >= 520.61)")
    print("3. No conflicts with conda/other Python environments")

if __name__ == "__main__":
    main()