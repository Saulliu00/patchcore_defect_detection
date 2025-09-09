#!/usr/bin/env python3
# evaluate_model.py - Standalone script to evaluate the trained model

import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from typing import Dict, List
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def evaluate_model_standalone():
    """Evaluate the trained model on test data"""
    
    print("=" * 60)
    print("📊 PatchCore Model Evaluation")
    print("=" * 60)
    
    # Check for model
    model_paths = [
        "models/deployment/patchcore_deployment.pth",
        "models/saved_models/patchcore_model.pth",
        "models/saved_models/patchcore_fixed/patchcore_model.pth",
        "models/saved_models/latest/patchcore_model.pth"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = Path(path)
            break
    
    if not model_path:
        print("❌ No trained model found!")
        print("   Please train a model first: python train.py")
        return
    
    print(f"✅ Found model: {model_path}")
    
    # Import inference module
    from inference.tiled_inference import TiledInference
    
    # Load model
    print("🔧 Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = TiledInference(
        model_path=str(model_path),
        tile_size=(256, 256),
        stride=(128, 128),
        device=device
    )
    print(f"✅ Model loaded on {device}")
    
    # Check for test data
    test_dirs = {
        "normal": Path("data/test/good"),
        "defective": Path("data/test/defective")
    }
    
    results = {
        "normal": {"correct": 0, "total": 0, "scores": []},
        "defective": {"correct": 0, "total": 0, "scores": []}
    }
    
    all_predictions = []
    
    # Process test images
    for category, test_dir in test_dirs.items():
        if not test_dir.exists():
            print(f"⚠️ No {category} test directory found: {test_dir}")
            continue
        
        # Get all image files
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        
        if not image_files:
            print(f"⚠️ No images found in {test_dir}")
            continue
        
        print(f"\n📁 Testing {category} images: {len(image_files)} files")
        print("-" * 40)
        
        for img_path in image_files:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️ Failed to load: {img_path.name}")
                continue
            
            # Run detection
            try:
                result = detector.predict(image, threshold=0.5)
                
                is_defective = result["is_anomaly"]
                anomaly_score = result["anomaly_score"]
                
                # Check if prediction is correct
                expected_defective = (category == "defective")
                is_correct = (is_defective == expected_defective)
                
                results[category]["total"] += 1
                results[category]["scores"].append(anomaly_score)
                if is_correct:
                    results[category]["correct"] += 1
                
                # Store for detailed analysis
                all_predictions.append({
                    'file': img_path.name,
                    'category': category,
                    'anomaly_score': anomaly_score,
                    'predicted': 'defective' if is_defective else 'normal',
                    'actual': category,
                    'correct': is_correct
                })
                
                # Display result
                status = "✅" if is_correct else "❌"
                predicted_label = "DEFECTIVE" if is_defective else "NORMAL"
                print(f"  {status} {img_path.name}: Score={anomaly_score:.4f}, Predicted={predicted_label}")
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")
                results[category]["total"] += 1
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    
    # Per-category results
    for category, stats in results.items():
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"]) * 100
            avg_score = np.mean(stats["scores"]) if stats["scores"] else 0
            
            print(f"\n{category.upper()} Images:")
            print(f"  Correct: {stats['correct']}/{stats['total']}")
            print(f"  Accuracy: {accuracy:.1f}%")
            print(f"  Avg Score: {avg_score:.4f}")
            
            if stats["scores"]:
                print(f"  Score Range: [{min(stats['scores']):.4f}, {max(stats['scores']):.4f}]")
    
    # Overall metrics
    total_correct = sum(r["correct"] for r in results.values())
    total_images = sum(r["total"] for r in results.values())
    
    if total_images > 0:
        overall_accuracy = (total_correct / total_images) * 100
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Images: {total_images}")
        print(f"  Correct Predictions: {total_correct}")
        print(f"  Overall Accuracy: {overall_accuracy:.1f}%")
        
        # Calculate additional metrics if we have both categories
        if results["normal"]["total"] > 0 and results["defective"]["total"] > 0:
            # True Positive Rate (Sensitivity) - correctly identifying defective
            tpr = (results["defective"]["correct"] / results["defective"]["total"]) * 100
            # True Negative Rate (Specificity) - correctly identifying normal
            tnr = (results["normal"]["correct"] / results["normal"]["total"]) * 100
            
            print(f"\nDETAILED METRICS:")
            print(f"  Sensitivity (TPR): {tpr:.1f}% (detecting defects)")
            print(f"  Specificity (TNR): {tnr:.1f}% (detecting normal)")
            
            # F1 Score
            if results["defective"]["correct"] > 0:
                precision = results["defective"]["correct"] / (
                    results["defective"]["correct"] + (results["normal"]["total"] - results["normal"]["correct"])
                )
                recall = results["defective"]["correct"] / results["defective"]["total"]
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1 Score: {f1:.3f}")
    
    # Save detailed results
    if all_predictions:
        results_df = pd.DataFrame(all_predictions)
        results_path = Path("evaluation_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Detailed results saved to: {results_path}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if total_images == 0:
        print("⚠️ No test images found! Add test images to:")
        print("   - data/test/good/ (normal images)")
        print("   - data/test/defective/ (defective images)")
    elif overall_accuracy < 70:
        print("⚠️ Low accuracy detected. Consider:")
        print("   1. Adding more training data (normal images)")
        print("   2. Ensuring consistent image quality")
        print("   3. Adjusting the detection threshold")
        print("   4. Checking if defects are visible enough")
    elif overall_accuracy < 90:
        print("✅ Good performance! To improve further:")
        print("   1. Fine-tune the detection threshold")
        print("   2. Add more diverse training samples")
        print("   3. Ensure consistent lighting/positioning")
    else:
        print("🎉 Excellent performance!")
        print("   Model is ready for deployment")
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    evaluate_model_standalone()