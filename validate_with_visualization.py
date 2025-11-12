#!/usr/bin/env python3
# validate_with_visualization.py - Enhanced PatchCore validation with per-image results and heatmaps

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from tqdm import tqdm
import yaml
import warnings
import csv
warnings.filterwarnings("ignore")

# Fix tensor core precision for RTX 3080 Ti
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

class VisualizingValidator:
    """PatchCore validator with per-image visualization and detailed results"""

    def __init__(self, model_path, device=None, config_path="config/patchcore_config.yaml"):
        self.model_path = Path(model_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None

        # Load configuration
        print(f"Loading configuration from {config_path}...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.image_size = tuple(config['dataset']['image_size'])
        print(f"  Image size: {self.image_size}")

        print(f"Initializing validator...")
        print(f"  Model: {self.model_path}")
        print(f"  Device: {self.device}")

        self._load_and_setup_model()
        self._setup_transforms()

    def _load_and_setup_model(self):
        """Load the PatchCore model"""
        from anomalib.models import Patchcore

        checkpoint = torch.load(self.model_path, map_location=self.device)

        config = {
            'backbone': checkpoint.get('backbone', 'resnet18'),
            'layers': checkpoint.get('layers', ['layer2', 'layer3']),
            'coreset_sampling_ratio': 0.1,
            'num_neighbors': 9
        }

        self.model = Patchcore(**config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    def _setup_transforms(self):
        """Setup image transformations"""
        from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
        import torch as torch_module

        self.transform = Compose([
            Resize(self.image_size),
            ToImage(),
            ToDtype(torch_module.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_single_image(self, image_path):
        """Predict on a single image and return score + heatmap"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size

            # Transform
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)

            # Extract score
            if hasattr(outputs, 'pred_score'):
                score = float(outputs.pred_score)
            elif hasattr(outputs, 'anomaly_score'):
                score = float(outputs.anomaly_score)
            elif isinstance(outputs, dict) and 'pred_score' in outputs:
                score = float(outputs['pred_score'])
            elif isinstance(outputs, dict) and 'anomaly_score' in outputs:
                score = float(outputs['anomaly_score'])
            else:
                score = 0.5

            # Extract heatmap (anomaly map)
            heatmap = None
            if hasattr(outputs, 'anomaly_map'):
                heatmap = outputs.anomaly_map.cpu().numpy().squeeze()
            elif isinstance(outputs, dict) and 'anomaly_map' in outputs:
                heatmap = outputs['anomaly_map'].cpu().numpy().squeeze()

            return {
                'score': score,
                'heatmap': heatmap,
                'original_size': original_size,
                'image': image
            }

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def validate_with_visualization(self, output_dir="validation_results"):
        """Run validation and save visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (output_dir / "visualizations" / "normal").mkdir(parents=True, exist_ok=True)
        (output_dir / "visualizations" / "defective").mkdir(parents=True, exist_ok=True)

        results = []

        # Process normal images
        test_good = Path("data/test/good")
        if test_good.exists():
            good_images = sorted(list(test_good.glob("*.jpg")) + list(test_good.glob("*.png")))
            print(f"\nProcessing {len(good_images)} normal images...")

            for img_path in tqdm(good_images, desc="Normal images"):
                result = self.predict_single_image(img_path)
                if result:
                    results.append({
                        'filename': img_path.name,
                        'category': 'normal',
                        'true_label': 'normal',
                        'score': result['score'],
                        'path': str(img_path)
                    })

                    # Save visualization
                    self._save_visualization(
                        result,
                        img_path.name,
                        result['score'],
                        'normal',
                        output_dir / "visualizations" / "normal" / f"{img_path.stem}_viz.png"
                    )

        # Process defective images
        test_defective = Path("data/test/defective")
        if test_defective.exists():
            defective_images = sorted(list(test_defective.glob("*.jpg")) + list(test_defective.glob("*.png")))
            print(f"\nProcessing {len(defective_images)} defective images...")

            for img_path in tqdm(defective_images, desc="Defective images"):
                result = self.predict_single_image(img_path)
                if result:
                    results.append({
                        'filename': img_path.name,
                        'category': 'defective',
                        'true_label': 'defective',
                        'score': result['score'],
                        'path': str(img_path)
                    })

                    # Save visualization
                    self._save_visualization(
                        result,
                        img_path.name,
                        result['score'],
                        'defective',
                        output_dir / "visualizations" / "defective" / f"{img_path.stem}_viz.png"
                    )

        # Calculate optimal threshold and metrics
        self._analyze_and_save_results(results, output_dir)

        return results

    def _save_visualization(self, result, filename, score, true_label, output_path):
        """Save visualization with original image and heatmap overlay"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(result['image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Heatmap
        if result['heatmap'] is not None:
            im = axes[1].imshow(result['heatmap'], cmap='jet', interpolation='bilinear')
            axes[1].set_title('Anomaly Heatmap')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)

            # Overlay
            axes[2].imshow(result['image'])
            overlay = axes[2].imshow(result['heatmap'], cmap='jet', alpha=0.5, interpolation='bilinear')
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            plt.colorbar(overlay, ax=axes[2], fraction=0.046)
        else:
            axes[1].text(0.5, 0.5, 'No heatmap available', ha='center', va='center')
            axes[1].axis('off')
            axes[2].text(0.5, 0.5, 'No heatmap available', ha='center', va='center')
            axes[2].axis('off')

        # Add title with score
        fig.suptitle(f'{filename} | True: {true_label} | Score: {score:.4f}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _analyze_and_save_results(self, results, output_dir):
        """Analyze results and save detailed reports"""
        print("\n" + "="*80)
        print("VALIDATION RESULTS WITH VISUALIZATION")
        print("="*80)

        # Separate normal and defective
        normal_results = [r for r in results if r['true_label'] == 'normal']
        defective_results = [r for r in results if r['true_label'] == 'defective']

        normal_scores = [r['score'] for r in normal_results]
        defective_scores = [r['score'] for r in defective_results]

        # Print score statistics
        if normal_scores:
            print(f"\nNormal images ({len(normal_scores)} images):")
            print(f"  Mean score: {np.mean(normal_scores):.4f}")
            print(f"  Min score: {np.min(normal_scores):.4f}")
            print(f"  Max score: {np.max(normal_scores):.4f}")
            print(f"  Std dev: {np.std(normal_scores):.4f}")

        if defective_scores:
            print(f"\nDefective images ({len(defective_scores)} images):")
            print(f"  Mean score: {np.mean(defective_scores):.4f}")
            print(f"  Min score: {np.min(defective_scores):.4f}")
            print(f"  Max score: {np.max(defective_scores):.4f}")
            print(f"  Std dev: {np.std(defective_scores):.4f}")

        # Find optimal threshold
        if normal_scores and defective_scores:
            all_scores = normal_scores + defective_scores
            all_labels = [0] * len(normal_scores) + [1] * len(defective_scores)

            best_threshold = None
            best_accuracy = 0

            for threshold in np.linspace(min(all_scores), max(all_scores), 100):
                predictions = [1 if s > threshold else 0 for s in all_scores]
                accuracy = sum(p == l for p, l in zip(predictions, all_labels)) / len(all_labels)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold

            print(f"\n📊 Optimal threshold: {best_threshold:.4f}")
            print(f"📊 Best accuracy: {best_accuracy:.2%}")

            # Add predictions to results
            for r in results:
                r['predicted_label'] = 'defective' if r['score'] > best_threshold else 'normal'
                r['correct'] = r['predicted_label'] == r['true_label']

            # Calculate detailed metrics
            tp = sum(1 for r in defective_results if r['score'] > best_threshold)
            fp = sum(1 for r in normal_results if r['score'] > best_threshold)
            tn = sum(1 for r in normal_results if r['score'] <= best_threshold)
            fn = sum(1 for r in defective_results if r['score'] <= best_threshold)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\n📈 Detailed metrics:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1 Score: {f1:.3f}")
            print(f"  True Positives: {tp}")
            print(f"  False Positives: {fp}")
            print(f"  True Negatives: {tn}")
            print(f"  False Negatives: {fn}")

            # Save detailed CSV report
            csv_path = output_dir / "detailed_results.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'category', 'true_label', 'predicted_label', 'score', 'correct', 'path'])
                writer.writeheader()
                writer.writerows(sorted(results, key=lambda x: x['score'], reverse=True))

            print(f"\n💾 Detailed results saved to: {csv_path}")

            # Print misclassified images
            incorrect = [r for r in results if not r['correct']]
            if incorrect:
                print(f"\n❌ Misclassified images ({len(incorrect)}):")
                for r in incorrect:
                    print(f"  {r['filename']:30s} | True: {r['true_label']:10s} | Predicted: {r['predicted_label']:10s} | Score: {r['score']:.4f}")
            else:
                print("\n✅ All images classified correctly!")

        print(f"\n📁 Visualizations saved to: {output_dir / 'visualizations'}")
        print("="*80)

def main():
    """Main function"""
    print("PatchCore Validation with Visualization")
    print("="*80)

    # Find model
    model_path = "models/minimal/patchcore_minimal.pth"

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"✅ Using model: {model_path}\n")

    # Create validator
    validator = VisualizingValidator(model_path)

    # Run validation with visualization
    results = validator.validate_with_visualization()

    print("\n✅ Validation complete!")
    print("\nNext steps:")
    print("1. Check 'validation_results/detailed_results.csv' for per-image scores")
    print("2. Review visualizations in 'validation_results/visualizations/'")
    print("3. Inspect misclassified images to understand model behavior")

if __name__ == "__main__":
    main()
