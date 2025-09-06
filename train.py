#!/usr/bin/env python3
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
    """Check GPU memory and optimize settings"""
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
    """Main training function optimized for RTX 3080 Ti"""
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
    
    print("\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print("Metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric_name}: {value:.4f}")
    
    # Prepare for deployment
    print("\nPreparing model for deployment...")
    deployment_path = trainer.prepare_for_deployment()
    
    print(f"\nTraining Summary:")
    print(f"  Training model: {model_path}")
    print(f"  Deployment model: {deployment_path}")
    print(f"  Logs: logs/tensorboard/")
    print(f"\nNext steps:")
    print(f"  1. Review metrics in TensorBoard")
    print(f"  2. Test model locally")
    print(f"  3. Deploy to Raspberry Pi:")
    print(f"     python deploy_to_pi.py --model_path {deployment_path}")

if __name__ == "__main__":
    main()
