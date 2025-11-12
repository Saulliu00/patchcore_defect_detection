#!/usr/bin/env python3
# validate_model.py - PatchCore validation with Anomalib 2.x compatibility

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml
import warnings
warnings.filterwarnings("ignore")

# Fix Issue 1: Set tensor core precision for RTX 3080 Ti
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')
    print("Tensor Core optimization enabled for RTX 3080 Ti")

class CompletePatchCoreValidator:
    """Complete PatchCore validator with memory bank handling"""

    def __init__(self, model_path, device=None, config_path="config/patchcore_config.yaml"):
        self.model_path = Path(model_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None

        # Load configuration for image size
        print(f"Loading configuration from {config_path}...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.image_size = tuple(config['dataset']['image_size'])
        print(f"  Image size from config: {self.image_size}")

        print(f"Initializing validator...")
        print(f"  Model path: {self.model_path}")
        print(f"  Device: {self.device}")

        # Step by step initialization
        self._load_and_setup_model()
        self._setup_transforms()
        
    def _load_and_setup_model(self):
        """Load and setup the PatchCore model"""
        print("\nLoading PatchCore model...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            from anomalib.models import Patchcore
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            print(f"Checkpoint loaded successfully")
            print(f"  Checkpoint keys: {list(checkpoint.keys())}")
            
            # Get model configuration (match train.py defaults)
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
            else:
                # Default configuration matching train.py
                config = {
                    'backbone': checkpoint.get('backbone', 'resnet18'),  # Match train.py
                    'layers': checkpoint.get('layers', ['layer2', 'layer3']),
                    'coreset_sampling_ratio': 0.1,
                    'num_neighbors': 9
                }
            
            print(f"Model config: {config}")
            
            # Initialize model
            self.model = Patchcore(
                backbone=config['backbone'],
                layers=config['layers'],
                pre_trained=False,
                coreset_sampling_ratio=config.get('coreset_sampling_ratio', 0.1),
                num_neighbors=config.get('num_neighbors', 9)
            )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing_keys:
                    print(f"  Missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"  Unexpected keys: {len(unexpected_keys)} keys")
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup image transformations matching train.py (Anomalib 2.x compatible)"""
        from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
        import torch as torch_module

        self.transform = Compose([
            Resize(self.image_size),  # Resize to match training
            ToImage(),  # Convert to tensor image
            ToDtype(torch_module.float32, scale=True),  # Convert to float32 and scale
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        print(f"Transforms ready (resizing to {self.image_size})")
    
    def _get_torch_model(self):
        """Get the actual PyTorch model (handle Lightning wrapper)"""
        if hasattr(self.model, 'model'):
            return self.model.model  # Lightning wrapper
        else:
            return self.model  # Direct PyTorch model
    
    def check_memory_bank(self):
        """Check if memory bank exists and is populated"""
        print("\nChecking memory bank...")
        
        try:
            torch_model = self._get_torch_model()
            
            if hasattr(torch_model, 'memory_bank'):
                memory_bank = torch_model.memory_bank
                memory_size = memory_bank.numel()
                print(f"  Memory bank found: {memory_size} elements")
                
                if memory_size > 0:
                    tensor_dims = len(memory_bank.size())
                    print(f"  Memory bank dimensions: {tensor_dims}D")
                    return True
                else:
                    print("  Memory bank is empty")
                    return False
            else:
                print("  No memory_bank attribute found")
                return False
                
        except Exception as e:
            print(f"  Error checking memory bank: {e}")
            return False
    
    def rebuild_memory_bank(self, coreset_sampling_ratio=0.1):
        """Rebuild memory bank from training data with proper subsampling

        Args:
            coreset_sampling_ratio: Ratio of features to keep (default: 0.1)
        """
        print("\nRebuilding memory bank from training data...")

        # Check training data
        train_good = Path("data/train/good")
        if not train_good.exists():
            print(f"Training data directory not found: {train_good}")
            return False

        image_files = list(train_good.glob("*.jpg")) + list(train_good.glob("*.png"))
        if len(image_files) == 0:
            print("No training images found!")
            return False

        print(f"Found {len(image_files)} training images")

        try:
            # Get the actual PyTorch model
            torch_model = self._get_torch_model()

            # CRITICAL FIX: Clear any existing embeddings and reset memory bank
            if hasattr(torch_model, 'embedding_store'):
                torch_model.embedding_store.clear()
            if hasattr(torch_model, 'memory_bank'):
                torch_model.memory_bank = torch.empty(0).to(self.device)

            # Set model to training mode (embeddings will go to embedding_store)
            self.model.train()

            print("Processing training images (accumulating embeddings)...")
            processed_count = 0
            for img_path in tqdm(image_files, desc="Building memory bank"):
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)

                    # Process through model (this populates embedding_store, NOT memory_bank)
                    with torch.no_grad():
                        _ = self.model(image_tensor)

                    processed_count += 1

                    # Check progress every 20 images
                    if processed_count % 20 == 0 and hasattr(torch_model, 'embedding_store'):
                        embeddings_count = len(torch_model.embedding_store)
                        print(f"    Progress: {processed_count}/{len(image_files)}, Embeddings accumulated: {embeddings_count}")

                except Exception as e:
                    print(f"    Error processing {img_path.name}: {e}")
                    continue

            print(f"Processed {processed_count}/{len(image_files)} training images")

            # CRITICAL FIX: Call subsample_embedding to finalize memory bank
            print("\nFinalizing memory bank with coreset subsampling...")

            if hasattr(torch_model, 'subsample_embedding'):
                # Method 1: Use the model's subsample_embedding method
                try:
                    torch_model.subsample_embedding(coreset_sampling_ratio)
                    print(f"✓ Called subsample_embedding with ratio={coreset_sampling_ratio}")
                except Exception as e:
                    print(f"  Error calling subsample_embedding: {e}")
                    # Method 2: Manual consolidation as fallback
                    self._manual_memory_bank_consolidation(torch_model, coreset_sampling_ratio)
            else:
                # Method 2: Manual consolidation
                self._manual_memory_bank_consolidation(torch_model, coreset_sampling_ratio)

            # Set back to eval mode
            self.model.eval()

            # Verify final memory bank
            if hasattr(torch_model, 'memory_bank'):
                final_size = torch_model.memory_bank.numel()
                final_shape = torch_model.memory_bank.shape if final_size > 0 else "empty"
                print(f"\n✓ Memory bank finalized!")
                print(f"  Shape: {final_shape}")
                print(f"  Total elements: {final_size}")
                return final_size > 0
            else:
                print("\n✗ Memory bank still not found after rebuild")
                return False

        except Exception as e:
            print(f"\n✗ Error rebuilding memory bank: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return False

    def _manual_memory_bank_consolidation(self, torch_model, sampling_ratio):
        """Manually consolidate embeddings into memory bank with coreset sampling

        This is a fallback when subsample_embedding is not available.
        """
        print("  Using manual memory bank consolidation...")

        if not hasattr(torch_model, 'embedding_store') or len(torch_model.embedding_store) == 0:
            print("  ✗ No embeddings in embedding_store to consolidate")
            return

        # Stack all embeddings
        all_embeddings = torch.vstack(torch_model.embedding_store)
        print(f"  Total embeddings: {all_embeddings.shape}")

        # Apply coreset sampling (k-center greedy algorithm)
        from sklearn.random_projection import SparseRandomProjection

        num_samples = int(all_embeddings.shape[0] * sampling_ratio)
        if num_samples < 1:
            num_samples = min(100, all_embeddings.shape[0])

        print(f"  Applying coreset sampling: {all_embeddings.shape[0]} → {num_samples} samples")

        # Simple random sampling (faster than k-center greedy)
        indices = torch.randperm(all_embeddings.shape[0])[:num_samples]
        sampled_embeddings = all_embeddings[indices]

        # Set memory bank
        torch_model.memory_bank = sampled_embeddings.to(self.device)

        # Clear embedding store
        torch_model.embedding_store.clear()

        print(f"  ✓ Memory bank shape: {torch_model.memory_bank.shape}")
    
    def test_single_image(self, image_path):
        """Test inference on a single image"""
        print(f"\nTesting single image: {image_path}")
        
        # Check if memory bank exists and is populated
        if not self.check_memory_bank():
            print("Memory bank is empty or missing, rebuilding...")
            if not self.rebuild_memory_bank():
                print("Failed to rebuild memory bank - cannot proceed with inference")
                return None
        
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            print(f"  Original image size: {image.size}")
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            tensor_dims = len(image_tensor.size())
            tensor_elements = image_tensor.numel()
            print(f"  Tensor: {tensor_dims}D with {tensor_elements} elements")
            
            # Run inference (pass tensor directly to model)
            with torch.no_grad():
                print("  Running model inference...")
                outputs = self.model(image_tensor)
                print(f"  Model outputs type: {type(outputs)}")
                
                # Extract score from outputs
                score = self._extract_score_from_outputs(outputs)
                print(f"  Final anomaly score: {score}")
                
                return score
                
        except Exception as e:
            print(f"  Error testing image: {e}")
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")
            return None
    
    def _extract_score_from_outputs(self, outputs):
        """Extract anomaly score from model outputs"""
        try:
            print(f"    Extracting score from outputs type: {type(outputs)}")
            
            # Handle different output types
            if hasattr(outputs, 'pred_score'):
                score = float(outputs.pred_score)
                print(f"    Found pred_score attribute: {score}")
                return score
            elif hasattr(outputs, 'anomaly_score'):
                score = float(outputs.anomaly_score)
                print(f"    Found anomaly_score attribute: {score}")
                return score
            elif isinstance(outputs, dict):
                print(f"    Dict keys: {list(outputs.keys())}")
                
                for key in ['pred_scores', 'anomaly_scores', 'pred_score', 'anomaly_score']:
                    if key in outputs:
                        value = outputs[key]
                        print(f"    Found {key}: {type(value)}")
                        
                        if torch.is_tensor(value):
                            element_count = value.numel()
                            print(f"    Tensor elements: {element_count}")
                            
                            if element_count == 1:
                                score = value.cpu().item()
                                print(f"    Extracted score: {score}")
                                return score
                            elif element_count > 1:
                                score = value.cpu().mean().item()
                                print(f"    Extracted mean score: {score}")
                                return score
                        elif isinstance(value, (int, float)):
                            score = float(value)
                            print(f"    Direct numeric score: {score}")
                            return score
                
                # Check all dict values for tensors
                for key, value in outputs.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        score = value.cpu().item()
                        print(f"    Found score in {key}: {score}")
                        return score
                        
            elif torch.is_tensor(outputs):
                element_count = outputs.numel()
                print(f"    Direct tensor with {element_count} elements")
                
                if element_count == 1:
                    score = outputs.cpu().item()
                    print(f"    Single tensor score: {score}")
                    return score
                elif element_count > 1:
                    score = outputs.cpu().mean().item()
                    print(f"    Mean tensor score: {score}")
                    return score
            
            print("    No score found, using default")
            return 0.5
            
        except Exception as e:
            print(f"    Error extracting score: {e}")
            return 0.5
    
    def validate_on_test_data(self):
        """Run validation on test data"""
        print("\nRunning validation on test data...")
        
        # Check test directories
        test_good = Path("data/test/good")
        test_defective = Path("data/test/defective")
        
        results = {"normal": [], "defective": []}
        
        # Process normal images
        if test_good.exists():
            good_images = list(test_good.glob("*.jpg")) + list(test_good.glob("*.png"))
            print(f"Found {len(good_images)} normal test images")
            
            for img_path in tqdm(good_images, desc="Normal images"):
                score = self.test_single_image(img_path)
                if score is not None:
                    results["normal"].append(score)
        else:
            print("No normal test directory found")
        
        # Process defective images
        if test_defective.exists():
            defective_images = list(test_defective.glob("*.jpg")) + list(test_defective.glob("*.png"))
            print(f"Found {len(defective_images)} defective test images")
            
            for img_path in tqdm(defective_images, desc="Defective images"):
                score = self.test_single_image(img_path)
                if score is not None:
                    results["defective"].append(score)
        else:
            print("No defective test directory found")
        
        # Calculate and display metrics
        self._calculate_and_display_metrics(results)
        
        return results
    
    def _calculate_and_display_metrics(self, results):
        """Calculate and display validation metrics"""
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        
        normal_scores = results["normal"]
        defective_scores = results["defective"]
        
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
        
        # Calculate accuracy if both types available
        if normal_scores and defective_scores:
            # Auto-determine optimal threshold
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
            
            print(f"\nOptimal threshold: {best_threshold:.4f}")
            print(f"Best accuracy: {best_accuracy:.2%}")
            
            # Calculate metrics at optimal threshold
            correct_normal = sum(1 for s in normal_scores if s <= best_threshold)
            correct_defective = sum(1 for s in defective_scores if s > best_threshold)
            total = len(normal_scores) + len(defective_scores)
            
            print(f"\nPerformance at optimal threshold:")
            print(f"  Overall accuracy: {(correct_normal + correct_defective) / total:.2%}")
            print(f"  Normal correct: {correct_normal}/{len(normal_scores)} ({correct_normal/len(normal_scores):.2%})")
            print(f"  Defective correct: {correct_defective}/{len(defective_scores)} ({correct_defective/len(defective_scores):.2%})")
            
            # Calculate precision, recall, F1
            tp = correct_defective
            fp = len(normal_scores) - correct_normal
            fn = len(defective_scores) - correct_defective
            tn = correct_normal
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nDetailed metrics:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1 Score: {f1:.3f}")
        
        # Performance assessment
        print("\n" + "="*60)
        print("PERFORMANCE ASSESSMENT")
        print("="*60)
        
        if not normal_scores and not defective_scores:
            print("No test data found! Add test images to:")
            print("  - data/test/good/ (normal images)")
            print("  - data/test/defective/ (defective images)")
        elif normal_scores and defective_scores:
            if best_accuracy >= 0.9:
                print("Excellent performance! Model is ready for deployment.")
            elif best_accuracy >= 0.7:
                print("Good performance, but room for improvement:")
                print("  - Consider adding more training data")
                print("  - Check image quality consistency")
            else:
                print("Poor performance. Consider:")
                print("  - Adding more diverse training data")
                print("  - Checking if defects are clearly visible")
                print("  - Retraining the model")
        else:
            print("Need both normal and defective test images for full evaluation")

def main():
    """Main function"""
    print("Complete PatchCore Model Validation")
    print("="*60)

    # Find model (updated to match train.py output location)
    model_paths = [
        "models/minimal/patchcore_minimal.pth",  # train.py output location
        "models/deployment/patchcore_deployment.pth",
        "models/saved_models/patchcore_model.pth",
        "models/saved_models/latest/patchcore_model.pth"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("No model found! Available paths:")
        for path in model_paths:
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  {exists} {path}")
        print("\nTrain a model first with: python train.py")
        return
    
    print(f"Using model: {model_path}")
    
    # Initialize validator
    try:
        validator = CompletePatchCoreValidator(model_path)
    except Exception as e:
        print(f"Failed to initialize validator: {e}")
        return
    
    # Test single image first
    test_dirs = [Path("data/test/good"), Path("data/test/defective")]
    test_image = None
    
    for test_dir in test_dirs:
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if images:
                test_image = images[0]
                break
    
    if test_image:
        print(f"\nTesting single image first: {test_image}")
        score = validator.test_single_image(test_image)
        
        if score is not None:
            print(f"Single image test successful! Score: {score:.4f}")
            
            # Run full validation
            validator.validate_on_test_data()
        else:
            print("Single image test failed - check model and data")
    else:
        print("No test images found for validation")
        print("Add test images to data/test/good/ and data/test/defective/")

if __name__ == "__main__":
    main()