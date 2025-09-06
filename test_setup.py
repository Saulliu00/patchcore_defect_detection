# test_setup.py - Simple script to test if everything is working
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import anomalib
        print(f"✅ Anomalib {anomalib.__version__}")
    except ImportError as e:
        print(f"❌ Anomalib import failed: {e}")
        print("Install with: pip install anomalib")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import lightning
        print(f"✅ Lightning {lightning.__version__}")
    except ImportError as e:
        print(f"❌ Lightning import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\nTesting project structure...")
    
    required_dirs = [
        "data/train/good",
        "data/test/good",
        "data/test/defective",
        "models/saved_models",
        "config",
        "src/training",
        "src/inference",
        "src/utils"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - missing")
            return False
    
    # Check for required files
    required_files = [
        "config/patchcore_config.yaml",
        "src/training/train_model.py",
        "src/training/config_handler.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - missing")
            return False
    
    return True

def test_data_availability():
    """Test if training data is available"""
    print("\nTesting data availability...")
    
    train_good = Path("data/train/good")
    if not train_good.exists():
        print("❌ Training data directory missing: data/train/good/")
        return False
    
    # Count images
    jpg_count = len(list(train_good.glob("*.jpg")))
    png_count = len(list(train_good.glob("*.png")))
    total_images = jpg_count + png_count
    
    print(f"📊 Training images found: {total_images} (JPG: {jpg_count}, PNG: {png_count})")
    
    if total_images == 0:
        print("❌ No training images found!")
        print("Place your normal/good part images in data/train/good/")
        return False
    elif total_images < 10:
        print("⚠️ Very few training images. Recommend at least 50+ for good results")
    
    return True

def test_config_file():
    """Test if config file is valid"""
    print("\nTesting configuration...")
    
    config_path = Path("config/patchcore_config.yaml")
    if not config_path.exists():
        print("❌ Config file missing: config/patchcore_config.yaml")
        return False
    
    try:
        # Add src to path for imports
        sys.path.append(str(Path(__file__).parent / "src"))
        from training.config_handler import ConfigHandler
        
        config_handler = ConfigHandler(config_path)
        config = config_handler.get_config()
        
        print(f"✅ Config loaded successfully")
        print(f"   Model: {config.model.name}")
        print(f"   Backbone: {config.model.backbone}")
        print(f"   Image size: {config.dataset.image_size}")
        print(f"   Batch size: {config.dataloader.train_batch_size}")
        
        return True
    except Exception as e:
        print(f"❌ Config file error: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing PatchCore Setup")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Training Data", test_data_availability),
        ("Configuration", test_config_file)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Ready for training: python train.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Fix the issues above before training")
    
    return all_passed

if __name__ == "__main__":
    main()