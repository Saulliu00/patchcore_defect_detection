#!/usr/bin/env python3
# test_model.py - Test trained model locally

import sys
from pathlib import Path
import cv2
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from inference.tiled_inference import TiledInference

def test_model():
    """Test the trained model on sample images"""
    
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
            
        print(f"\nTesting on {test_dir}...")
        
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
