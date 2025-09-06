# test_anomalib.py
def test_anomalib_import():
    print("Testing Anomalib imports...")
    
    try:
        # Test basic import
        import anomalib
        print(f"✅ Anomalib version: {anomalib.__version__}")
        
        # Test PatchCore import
        from anomalib.models.image.patchcore import Patchcore
        print("✅ PatchCore import successful")
        
        # Test data import
        from anomalib.data import Folder
        print("✅ Folder import successful")
        
        # Test engine import
        from anomalib.engine import Engine
        print("✅ Engine import successful")
        
        print("\n🎉 All Anomalib imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_anomalib_import()