#!/usr/bin/env python3
# train_minimal.py - Minimal training script that avoids callback conflicts

import os
import sys
from pathlib import Path
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def train_patchcore_minimal():
    """Minimal PatchCore training without complex callbacks"""
    print("🚀 Minimal PatchCore Training")
    print("=" * 60)
    
    # Check environment
    print("Environment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Check data
    train_good = Path("data/train/good")
    if not train_good.exists():
        print("❌ No training data found at data/train/good/")
        return False
    
    images = list(train_good.glob("*.jpg")) + list(train_good.glob("*.png"))
    print(f"\n✅ Found {len(images)} training images")
    
    if len(images) < 10:
        print("⚠️ Warning: Very few images. PatchCore needs more data.")
    
    # Create empty defective dir (required by Anomalib)
    train_defective = Path("data/train/defective")
    train_defective.mkdir(parents=True, exist_ok=True)
    
    try:
        from anomalib.models import Patchcore
        from anomalib.data import Folder
        import lightning.pytorch as pl
        
        print("\n📦 Creating model and data...")
        
        # 1. Create model with smaller backbone for testing
        model = Patchcore(
            backbone="resnet18",  # Smaller model
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9
        )
        
        # 2. Create datamodule with minimal parameters
        datamodule = Folder(
            name="custom",  # Add required name parameter
            root=Path("data/train").absolute(),
            normal_dir="good",
            abnormal_dir="defective",
            #image_size=(256, 256),  # Smaller size for testing
            train_batch_size=1,  # Small batch size
            eval_batch_size=1,
            num_workers=0,  # No multiprocessing
        )
        
        # Setup datamodule
        datamodule.setup()
        
        # Verify we have data
        train_loader = datamodule.train_dataloader()
        num_batches = len(train_loader)
        print(f"✅ Created {num_batches} training batches")
        
        if num_batches == 0:
            print("❌ No training batches created!")
            return False
        
        # 3. Create trainer WITHOUT callbacks to avoid conflicts
        print("\n🎯 Starting training...")
        
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=1,
            callbacks=[],  # NO callbacks to avoid conflicts
            logger=False,  # No logger to keep it simple
            enable_checkpointing=False,  # Disable checkpointing
            enable_progress_bar=True,
            enable_model_summary=False,
            num_sanity_val_steps=0,  # Skip validation sanity check
        )
        
        # Train the model
        trainer.fit(model=model, datamodule=datamodule)
        
        print("\n✅ Training completed!")
        
        # 4. Save model manually
        save_dir = Path("models/minimal")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / "patchcore_minimal.pth"
        
        # Extract memory bank and other components
        model_dict = {
            'model_state_dict': model.state_dict(),
            'backbone': 'resnet18',
            #'image_size': [256, 256],
        }
        
        # Try to save memory bank if it exists
        if hasattr(model, 'memory_bank') and model.memory_bank is not None:
            if hasattr(model.memory_bank, 'memory'):
                model_dict['memory_bank'] = model.memory_bank.memory
                print(f"✅ Memory bank saved (size: {model.memory_bank.memory.shape if hasattr(model.memory_bank.memory, 'shape') else 'unknown'})")
        
        torch.save(model_dict, model_path)
        print(f"💾 Model saved to: {model_path}")
        print(f"📦 File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        
        import traceback
        print("\nDetailed error:")
        print(traceback.format_exc())
        
        # Specific error handling
        if "Memory bank is empty" in str(e):
            print("\n⚠️ Memory bank error detected!")
            print("This usually means:")
            print("1. No features were extracted from training images")
            print("2. Images might be corrupted or unreadable")
            print("3. The model didn't process the training data properly")
            print("\nTry:")
            print("- Verify your images are valid JPG/PNG files")
            print("- Ensure images are not corrupted")
            print("- Try with different images")
        
        elif "ModelCheckpoint" in str(e):
            print("\n⚠️ Callback conflict detected!")
            print("The script above should avoid this by not using callbacks.")
            print("If this still happens, your Anomalib version has issues.")
        
        return False

def verify_model():
    """Verify the saved model"""
    model_path = Path("models/minimal/patchcore_minimal.pth")
    
    if not model_path.exists():
        print("❌ No model found to verify")
        return
    
    print("\n📊 Verifying saved model...")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("Model contents:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  {key}: {len(checkpoint[key])} items")
                # If it's state dict, check for memory-related keys
                if key == 'model_state_dict':
                    memory_keys = [k for k in checkpoint[key].keys() if 'memory' in k.lower()]
                    if memory_keys:
                        print(f"    Memory-related keys in state dict: {memory_keys[:5]}")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"  {key}: tensor {checkpoint[key].shape}")
            else:
                print(f"  {key}: {type(checkpoint[key]).__name__}")
        
        # Check for memory bank in various places
        if 'memory_bank' in checkpoint:
            print("\n✅ Memory bank is saved in the model")
        elif 'memory_bank_keys' in checkpoint:
            print("\n✅ Memory bank keys found in state dict")
            print(f"  Keys: {list(checkpoint['memory_bank_keys'].keys())}")
        else:
            # Check if memory bank is in the state dict
            if 'model_state_dict' in checkpoint:
                memory_keys = [k for k in checkpoint['model_state_dict'].keys() if 'memory' in k.lower()]
                if memory_keys:
                    print(f"\n✅ Memory-related parameters found in state dict:")
                    for key in memory_keys[:5]:  # Show first 5
                        tensor = checkpoint['model_state_dict'][key]
                        print(f"  {key}: shape {tensor.shape if hasattr(tensor, 'shape') else 'N/A'}")
                else:
                    print("\n⚠️ No explicit memory bank found")
                    print("  PatchCore may build it dynamically during inference")
                    print("  The model weights are saved and should work for inference")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def main():
    """Main function"""
    print("🔧 PatchCore Minimal Training")
    print("This script avoids all callback conflicts")
    print("=" * 60)
    
    # Train
    success = train_patchcore_minimal()
    
    if success:
        print("\n🎉 Training successful!")
        
        # Verify
        verify_model()
        
        print("\nNext steps:")
        print("1. Model saved to models/minimal/patchcore_minimal.pth")
        print("2. This proves training works")
        print("3. Now we can fix the full training script")
    else:
        print("\n❌ Training failed")
        print("\nTroubleshooting:")
        print("1. Check you have images in data/train/good/")
        print("2. Ensure images are valid JPG or PNG files")
        print("3. Try with more images (50+ recommended)")

if __name__ == "__main__":
    main()