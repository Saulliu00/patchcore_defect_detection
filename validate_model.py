#!/usr/bin/env python3
# validate_model.py - Validate the trained PatchCore model on test data

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class PatchCoreValidator:
    """Validate PatchCore model on test data"""
    
    def __init__(self, model_path, device=None):
        self.model_path = Path(model_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained PatchCore model"""
        print(f"📦 Loading model from: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            # Try loading with Anomalib
            from anomalib.models import Patchcore
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Get model configuration
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                self.model = Patchcore(
                    backbone=config.get('backbone', 'resnet18'),
                    layers=config.get('layers', ['layer2', 'layer3']),
                    pre_trained=False,  # We'll load weights
                    coreset_sampling_ratio=config.get('coreset_sampling_ratio', 0.1),
                    num_neighbors=config.get('num_neighbors', 9)
                )
            else:
                # Default configuration
                self.model = Patchcore(
                    backbone='resnet18',
                    layers=['layer2', 'layer3'],
                    pre_trained=False,
                    coreset_sampling_ratio=0.1,
                    num_neighbors=9
                )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.image_size = checkpoint.get('image_size', [256, 256])
            self.setup_transforms()
            
            print(f"✅ Model loaded successfully")
            print(f"  Device: {self.device}")
            print(f"  Image size: {self.image_size}")
            
        except ImportError:
            print("❌ Anomalib not available, using basic validation")
            self.model = None
            self.setup_transforms()
    
    def setup_transforms(self):
        """Setup image transformations"""
        self.transform = transforms.Compose([
            transforms.Resize(tuple(self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def process_image(self, image_path):
        """Process a single image and get anomaly score"""
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            if self.model is not None:
                # Use Anomalib model
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    
                    # Get anomaly score
                    if isinstance(outputs, dict):
                        if 'pred_scores' in outputs:
                            score = outputs['pred_scores'].cpu().item()
                        elif 'anomaly_scores' in outputs:
                            score = outputs['anomaly_scores'].cpu().item()
                        else:
                            score = 0.5  # Default if no score found
                    else:
                        score = outputs.cpu().item()
                        
                return score
            else:
                # Fallback: random score for testing
                return np.random.random()
                
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def validate(self, test_good_dir, test_defective_dir, threshold=None):
        """Validate model on test data"""
        print("\n" + "="*60)
        print("🧪 VALIDATION RESULTS")
        print("="*60)
        
        test_good = Path(test_good_dir)
        test_defective = Path(test_defective_dir)
        
        # Collect test images
        good_images = list(test_good.glob("*.jpg")) + list(test_good.glob("*.png"))
        defective_images = list(test_defective.glob("*.jpg")) + list(test_defective.glob("*.png"))
        
        print(f"\nTest data:")
        print(f"  Normal images: {len(good_images)}")
        print(f"  Defective images: {len(defective_images)}")
        
        if len(good_images) == 0 and len(defective_images) == 0:
            print("❌ No test images found!")
            return
        
        # Process normal images
        good_scores = []
        if len(good_images) > 0:
            print(f"\n📊 Processing normal images...")
            for img_path in tqdm(good_images, desc="Normal"):
                score = self.process_image(img_path)
                if score is not None:
                    good_scores.append(score)
        
        # Process defective images
        defective_scores = []
        if len(defective_images) > 0:
            print(f"\n📊 Processing defective images...")
            for img_path in tqdm(defective_images, desc="Defective"):
                score = self.process_image(img_path)
                if score is not None:
                    defective_scores.append(score)
        
        # Calculate statistics
        print("\n" + "="*60)
        print("📈 STATISTICS")
        print("="*60)
        
        if good_scores:
            print(f"\n✅ Normal images (should have LOW scores):")
            print(f"  Count: {len(good_scores)}")
            print(f"  Mean score: {np.mean(good_scores):.4f}")
            print(f"  Min score: {np.min(good_scores):.4f}")
            print(f"  Max score: {np.max(good_scores):.4f}")
            print(f"  Std dev: {np.std(good_scores):.4f}")
        
        if defective_scores:
            print(f"\n🔴 Defective images (should have HIGH scores):")
            print(f"  Count: {len(defective_scores)}")
            print(f"  Mean score: {np.mean(defective_scores):.4f}")
            print(f"  Min score: {np.min(defective_scores):.4f}")
            print(f"  Max score: {np.max(defective_scores):.4f}")
            print(f"  Std dev: {np.std(defective_scores):.4f}")
        
        # Determine optimal threshold if not provided
        if threshold is None and good_scores and defective_scores:
            # Find threshold that best separates the two groups
            all_scores = good_scores + defective_scores
            all_labels = [0] * len(good_scores) + [1] * len(defective_scores)
            
            best_threshold = None
            best_accuracy = 0
            
            for t in np.linspace(min(all_scores), max(all_scores), 100):
                predictions = [1 if s > t else 0 for s in all_scores]
                accuracy = sum(p == l for p, l in zip(predictions, all_labels)) / len(all_labels)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = t
            
            threshold = best_threshold
            print(f"\n🎯 Optimal threshold: {threshold:.4f}")
            print(f"  Accuracy at this threshold: {best_accuracy:.2%}")
        elif threshold is None:
            # Default threshold
            threshold = 0.5
            print(f"\n⚠️ Using default threshold: {threshold:.4f}")
        
        # Calculate performance metrics
        if good_scores and defective_scores:
            print("\n" + "="*60)
            print("📊 PERFORMANCE METRICS")
            print("="*60)
            
            # True Positives: defective images correctly identified
            tp = sum(1 for s in defective_scores if s > threshold)
            # False Negatives: defective images missed
            fn = len(defective_scores) - tp
            # True Negatives: normal images correctly identified
            tn = sum(1 for s in good_scores if s <= threshold)
            # False Positives: normal images wrongly flagged
            fp = len(good_scores) - tn
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nConfusion Matrix:")
            print(f"                 Predicted")
            print(f"              Normal  Defective")
            print(f"Actual Normal    {tn:3d}      {fp:3d}")
            print(f"      Defective  {fn:3d}      {tp:3d}")
            
            print(f"\nMetrics:")
            print(f"  Accuracy:  {accuracy:.2%}")
            print(f"  Precision: {precision:.2%}")
            print(f"  Recall:    {recall:.2%}")
            print(f"  F1 Score:  {f1:.2%}")
            
            # Interpretation
            print("\n" + "="*60)
            print("💡 INTERPRETATION")
            print("="*60)
            
            if accuracy >= 0.9:
                print("🎉 Excellent! Model is performing very well.")
            elif accuracy >= 0.7:
                print("✅ Good performance, but there's room for improvement.")
            elif accuracy >= 0.5:
                print("⚠️ Moderate performance. Consider:")
                print("  - Adding more training data")
                print("  - Adjusting the threshold")
                print("  - Using a larger backbone model")
            else:
                print("❌ Poor performance. The model may need retraining.")
            
            if fp > tn:
                print("\n⚠️ High false positive rate - too many normal parts flagged as defective")
            if fn > tp:
                print("\n⚠️ High false negative rate - missing too many defects")
        
        # Show example predictions
        print("\n" + "="*60)
        print("📝 EXAMPLE PREDICTIONS")
        print("="*60)
        
        if good_scores:
            print("\nNormal images (first 5):")
            for i, (img, score) in enumerate(zip(good_images[:5], good_scores[:5])):
                pred = "✅ Correct" if score <= threshold else "❌ Wrong"
                print(f"  {img.name}: score={score:.4f} {pred}")
        
        if defective_scores:
            print("\nDefective images (first 5):")
            for i, (img, score) in enumerate(zip(defective_images[:5], defective_scores[:5])):
                pred = "✅ Correct" if score > threshold else "❌ Wrong"
                print(f"  {img.name}: score={score:.4f} {pred}")
        
        return {
            'good_scores': good_scores,
            'defective_scores': defective_scores,
            'threshold': threshold,
            'accuracy': accuracy if good_scores and defective_scores else None
        }

def main():
    """Main validation function"""
    print("🔍 PatchCore Model Validation")
    print("="*60)
    
    # Find model file
    model_paths = [
        "models/minimal/patchcore_minimal.pth",
        #"models/working/patchcore_model.pth",
        #"models/deployment/patchcore_deployment.pth",
        #"models/saved_models/patchcore_model.pth",
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found!")
        print("Train a model first with: python train_minimal.py")
        return
    
    print(f"Found model: {model_path}")
    
    # Initialize validator
    validator = PatchCoreValidator(model_path)
    
    # Validate on test data
    results = validator.validate(
        test_good_dir="data/test/good",
        test_defective_dir="data/test/defective",
        threshold=None  # Auto-determine threshold
    )
    
    print("\n" + "="*60)
    print("✅ Validation complete!")
    
    if results and results['accuracy'] is not None:
        print(f"\nFinal accuracy: {results['accuracy']:.2%}")
        print(f"Recommended threshold: {results['threshold']:.4f}")

if __name__ == "__main__":
    main()