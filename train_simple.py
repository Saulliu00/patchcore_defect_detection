#!/usr/bin/env python3
# train_simple.py - Simplified training script for normal-only data

import os
import sys
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from training.train_model import PatchCoreTrainer
from training.config_handler import ConfigHandler

def main():
    """Main training function - simplified for normal-only training"""
    print("=" * 60)
    print("🚀 PatchCore Training - Normal Data Only Mode")
    print("=" * 60)
    print()
    print("ℹ️  PatchCore Information:")
    print("  • Trains ONLY on normal/good samples")
    print("  • Does NOT require defective samples for training")
    print("  • Learns what 'normal' looks like")
    print("  • Detects anomalies as deviations from normal")
    print()
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Detected: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  No GPU detected - Training will use CPU (slower)")
    
    print()
    print("📁 Checking training data...")
    
    # Check data directories
    data_path = Path("data")
    train_good = data_path / "train" / "good"
    
    # Only check for normal training data
    if not train_good.exists():
        print("❌ ERROR: Training directory not found!")
        print(f"   Expected path: {train_good}")
        print()
        print("📝 Instructions:")
        print("   1. Create directory: data/train/good/")
        print("   2. Place your NORMAL part images there")
        print("   3. You need at least 50+ normal images")
        print("   4. Do NOT create data/train/defective/")
        return
    
    # Count training images
    image_files = list(train_good.glob("*.jpg")) + list(train_good.glob("*.png"))
    num_images = len(image_files)
    
    if num_images == 0:
        print("❌ ERROR: No images found in data/train/good/")
        print("   Please add normal part images (JPG or PNG)")
        return
    
    print(f"✅ Found {num_images} normal training images")
    
    if num_images < 50:
        print(f"⚠️  Warning: Only {num_images} images found")
        print("   Recommended: 50+ images for better results")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Optional: Check for test data (not required)
    test_good = data_path / "test" / "good"
    test_defective = data_path / "test" / "defective"
    
    has_test_data = False
    if test_good.exists() or test_defective.exists():
        test_good_count = len(list(test_good.glob("*.jpg")) + list(test_good.glob("*.png"))) if test_good.exists() else 0
        test_defective_count = len(list(test_defective.glob("*.jpg")) + list(test_defective.glob("*.png"))) if test_defective.exists() else 0
        
        if test_good_count > 0 or test_defective_count > 0:
            print()
            print("📊 Test data found (optional):")
            print(f"   Normal test images: {test_good_count}")
            print(f"   Defective test images: {test_defective_count}")
            has_test_data = True
    
    if not has_test_data:
        print()
        print("ℹ️  No test data found - this is OK!")
        print("   Training will proceed without evaluation")
        print("   You can test the model after training")
    
    print()
    print("=" * 60)
    
    # Load configuration
    print("📋 Loading configuration...")
    config_handler = ConfigHandler("config/patchcore_config.yaml")
    config = config_handler.get_config()
    
    # Auto-adjust settings based on GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 10:  # High-end GPU
            config.dataloader.train_batch_size = 16
            config.dataloader.num_workers = 8
            config.dataset.image_size = [512, 512]  # Higher resolution
            print("✅ Using high-performance GPU settings")
        elif gpu_memory >= 6:  # Mid-range GPU
            config.dataloader.train_batch_size = 8
            config.dataloader.num_workers = 4
            config.dataset.image_size = [256, 256]
            print("✅ Using standard GPU settings")
        else:  # Low-end GPU
            config.dataloader.train_batch_size = 4
            config.dataloader.num_workers = 2
            config.dataset.image_size = [256, 256]
            print("✅ Using conservative GPU settings")
    else:
        # CPU settings
        config.dataloader.train_batch_size = 2
        config.dataloader.num_workers = 0
        config.dataset.image_size = [256, 256]
        print("✅ Using CPU settings")
    
    print(f"   Batch size: {config.dataloader.train_batch_size}")
    print(f"   Image size: {config.dataset.image_size}")
    print(f"   Workers: {config.dataloader.num_workers}")
    
    # Initialize trainer
    print()
    print("🔧 Initializing trainer...")
    trainer = PatchCoreTrainer(config)
    
    # Train the model
    print()
    print("🎯 Starting training...")
    print("   This should take a few minutes...")
    print("   Monitor with: tensorboard --logdir logs/tensorboard")
    print()
    
    try:
        model_path = trainer.train()
        print(f"\n✅ Model saved to: {model_path}")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have only normal images in data/train/good/")
        print("2. Check that images are valid JPG/PNG files")
        print("3. Try reducing batch size in config/patchcore_config.yaml")
        return
    
    # Evaluate if test data exists
    if has_test_data:
        print("\n📊 Evaluating model...")
        metrics = trainer.evaluate()
        
        if metrics and not metrics.get('error'):
            print("\n📈 Evaluation Metrics:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric_name}: {value:.4f}")
    
    # Prepare for deployment
    print("\n📦 Preparing for deployment...")
    deployment_path = trainer.prepare_for_deployment()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("📁 Output Files:")
    print(f"   Training model: {model_path}")
    print(f"   Deployment model: {deployment_path}")
    print(f"   Logs: logs/tensorboard/")
    print()
    print("📝 Next Steps:")
    print("   1. Test locally: python test_model.py")
    print("   2. View metrics: tensorboard --logdir logs/tensorboard")
    print("   3. Deploy to Pi: python deploy_to_pi.py --model_path " + str(deployment_path))
    print()
    print("✅ Your model is ready for defect detection!")

if __name__ == "__main__":
    main()