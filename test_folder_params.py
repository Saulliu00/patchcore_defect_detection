# test_folder_params.py - Check what parameters Folder accepts
import inspect
from anomalib.data import Folder

def test_folder_parameters():
    """Test what parameters the Folder class accepts"""
    print("🔍 Checking Folder class parameters...")
    
    # Get the signature of Folder.__init__
    signature = inspect.signature(Folder.__init__)
    params = list(signature.parameters.keys())
    
    print("Available parameters for Folder class:")
    for i, param in enumerate(params, 1):
        if param != 'self':
            param_info = signature.parameters[param]
            default = param_info.default
            required = default == inspect.Parameter.empty
            status = "REQUIRED" if required else f"default={default}"
            print(f"  {i}. {param} ({status})")
    
    print(f"\nTotal parameters: {len(params) - 1}")  # Subtract 'self'
    
    # Try to create a minimal Folder instance
    print("\n🧪 Testing minimal Folder creation...")
    
    try:
        folder = Folder(
            name="test_folder",  # Add required name parameter
            root="./data",
            normal_dir="good",
            abnormal_dir="defective"
        )
        print("✅ Minimal Folder creation successful")
        
        # Check if it has specific attributes
        attrs_to_check = ['image_size', 'train_batch_size', 'eval_batch_size', 'num_workers']
        print("\nChecking available attributes:")
        for attr in attrs_to_check:
            if hasattr(folder, attr):
                print(f"  ✅ {attr}: {getattr(folder, attr, 'N/A')}")
            else:
                print(f"  ❌ {attr}: Not available")
                
    except Exception as e:
        print(f"❌ Minimal creation failed: {e}")
        
    # Try with more parameters
    print("\n🧪 Testing Folder with common parameters...")
    try:
        folder_full = Folder(
            name="test_folder_full",
            root="./data", 
            normal_dir="good",
            abnormal_dir="defective",
            image_size=(256, 256),
            train_batch_size=8,
            eval_batch_size=8,
            num_workers=4
        )
        print("✅ Full Folder creation successful")
    except Exception as e:
        print(f"❌ Full creation failed: {e}")
        print("Trying to identify unsupported parameters...")
        
        # Test individual parameters
        base_params = {
            "name": "test",
            "root": "./data",
            "normal_dir": "good", 
            "abnormal_dir": "defective"
        }
        
        test_params = {
            "image_size": (256, 256),
            "train_batch_size": 8,
            "eval_batch_size": 8, 
            "num_workers": 4,
            "center_crop": None,
            "normalization": "imagenet"
        }
        
        working_params = base_params.copy()
        
        for param_name, param_value in test_params.items():
            try:
                test_kwargs = working_params.copy()
                test_kwargs[param_name] = param_value
                Folder(**test_kwargs)
                working_params[param_name] = param_value
                print(f"  ✅ {param_name}: Works")
            except Exception as e:
                print(f"  ❌ {param_name}: Failed - {e}")

if __name__ == "__main__":
    test_folder_parameters()