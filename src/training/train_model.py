# src/training/train_model.py - Fixed for proper Anomalib path handling
import os
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# Import with compatibility handling
try:
    # Try Anomalib imports
    from anomalib.data import Folder
    from anomalib.engine import Engine
    try:
        from anomalib.models.image.patchcore import Patchcore
        print("✅ Using Anomalib 1.0+ image-specific imports")
    except ImportError:
        from anomalib.models import Patchcore
        print("✅ Using Anomalib 1.0+ general imports")
except ImportError:
    print("❌ Anomalib not installed properly")
    print("Run: pip install anomalib>=1.0.0")
    raise

from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
import inspect

class PatchCoreTrainer:
    """PatchCore model trainer - fixed for proper Anomalib path handling"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.datamodule = None
        self.engine = None
        self.setup_model()
        self.setup_data()
        
    def setup_model(self):
        """Initialize PatchCore model with error handling"""
        print("🔧 Initializing PatchCore model...")
        
        try:
            # Initialize with all required parameters
            self.model = Patchcore(
                backbone=self.config.model.backbone,
                layers=self.config.model.layers,
                pre_trained=self.config.model.pre_trained,
                coreset_sampling_ratio=self.config.model.coreset_sampling_ratio,
                num_neighbors=self.config.model.num_neighbors
            )
            print(f"✅ Model initialized with backbone: {self.config.model.backbone}")
            
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            # Try with minimal parameters
            try:
                self.model = Patchcore()
                print("✅ Model initialized with default parameters")
            except Exception as e2:
                print(f"❌ Critical error: {e2}")
                raise
        
    def setup_data(self):
        """Setup data module with validation - handles normal-only training data"""
        print("📊 Setting up data module...")
        
        # Validate data directory structure
        data_path = Path(self.config.dataset.root)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
            
        # Check for training data (only normal data required)
        train_path = data_path / "train"
        train_good_path = train_path / self.config.dataset.normal_dir
        
        if not train_good_path.exists():
            raise FileNotFoundError(f"Training normal data not found: {train_good_path}")
            
        # Count training images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        training_images = []
        for ext in image_extensions:
            training_images.extend(list(train_good_path.glob(ext)))
            training_images.extend(list(train_good_path.glob(ext.upper())))
            
        if len(training_images) == 0:
            raise ValueError(f"No training images found in {train_good_path}")
            
        print(f"✅ Found {len(training_images)} training images (normal/good)")
        
        # Check if defective training data exists (it shouldn't for PatchCore)
        train_defective_path = train_path / self.config.dataset.abnormal_dir
        if train_defective_path.exists():
            defective_count = len(list(train_defective_path.glob("*.[jp][pn]g")))
            if defective_count > 0:
                print(f"⚠️ Warning: Found {defective_count} defective images in training data")
                print("⚠️ PatchCore trains on normal data only - defective training data will be ignored")
        
        # Validate test data
        test_path = data_path / "test"
        test_good_path = test_path / self.config.dataset.normal_dir
        test_defective_path = test_path / self.config.dataset.abnormal_dir
        
        test_good_count = 0
        test_defective_count = 0
        
        if test_good_path.exists():
            test_good_images = []
            for ext in image_extensions:
                test_good_images.extend(list(test_good_path.glob(ext)))
                test_good_images.extend(list(test_good_path.glob(ext.upper())))
            test_good_count = len(test_good_images)
        
        if test_defective_path.exists():
            test_defective_images = []
            for ext in image_extensions:
                test_defective_images.extend(list(test_defective_path.glob(ext)))
                test_defective_images.extend(list(test_defective_path.glob(ext.upper())))
            test_defective_count = len(test_defective_images)
        
        print(f"✅ Found {test_good_count} normal test images")
        print(f"✅ Found {test_defective_count} defective test images")
        
        if test_good_count == 0:
            print("⚠️ Warning: No normal test images found - evaluation will be limited")
        if test_defective_count == 0:
            print("⚠️ Warning: No defective test images found - evaluation will be limited")
        
        # Setup Folder datamodule with correct path handling
        self._setup_datamodule_with_correct_paths(data_path)
        
    def _setup_datamodule_with_correct_paths(self, data_path: Path):
        """Setup datamodule with correct path handling for Anomalib"""
        
        # Get Folder class signature to detect available parameters
        folder_sig = inspect.signature(Folder.__init__)
        folder_params = list(folder_sig.parameters.keys())
        
        print(f"📝 Detected Folder parameters: {folder_params}")
        
        # CRITICAL: Create the expected directory structure for Anomalib
        # Anomalib expects the structure: root/normal_dir and root/abnormal_dir
        # But our data is in: data/train/good and data/test/good, data/test/defective
        
        # Strategy 1: Try with train as root
        train_root = data_path / "train"
        
        print(f"🔍 Trying Strategy 1: Using {train_root} as root")
        print(f"   Looking for: {train_root / self.config.dataset.normal_dir}")
        
        # Build parameters for Strategy 1
        params = {
            'root': train_root,
        }
        
        # Add name if required
        if 'name' in folder_params:
            params['name'] = self.config.dataset.name
            
        # Add directories
        if 'normal_dir' in folder_params:
            params['normal_dir'] = self.config.dataset.normal_dir  # "good"
        if 'abnormal_dir' in folder_params:
            params['abnormal_dir'] = self.config.dataset.abnormal_dir  # "defective"
            
        # Add optional parameters
        optional_params = {
            'train_batch_size': self.config.dataloader.train_batch_size,
            'eval_batch_size': self.config.dataloader.eval_batch_size,
            'num_workers': self.config.dataloader.num_workers,
            'task': self.config.dataset.task if hasattr(self.config.dataset, 'task') else 'segmentation',
            'image_size': tuple(self.config.dataset.image_size) if hasattr(self.config.dataset, 'image_size') else (256, 256),
            'seed': 42
        }
        
        # Only add parameters that are actually accepted
        for param_name, param_value in optional_params.items():
            if param_name in folder_params:
                params[param_name] = param_value
        
        print(f"📦 Strategy 1 - Creating Folder datamodule with parameters: {list(params.keys())}")
        
        try:
            self.datamodule = Folder(**params)
            print("✅ Strategy 1 successful - Data module initialized")
            return
            
        except Exception as e1:
            print(f"❌ Strategy 1 failed: {e1}")
            
            # Strategy 2: Try creating symlinks or copying data to expected structure
            print(f"🔍 Trying Strategy 2: Creating expected structure")
            
            # Create temporary structure that Anomalib expects
            temp_train_root = data_path / "anomalib_train"
            temp_train_root.mkdir(exist_ok=True)
            
            # Create symlinks to actual data
            temp_good_link = temp_train_root / self.config.dataset.normal_dir
            temp_defective_link = temp_train_root / self.config.dataset.abnormal_dir
            
            # Remove existing symlinks if they exist
            if temp_good_link.exists():
                if temp_good_link.is_symlink():
                    temp_good_link.unlink()
                elif temp_good_link.is_dir():
                    import shutil
                    shutil.rmtree(temp_good_link)
            
            if temp_defective_link.exists():
                if temp_defective_link.is_symlink():
                    temp_defective_link.unlink()
                elif temp_defective_link.is_dir():
                    import shutil
                    shutil.rmtree(temp_defective_link)
            
            # Create symlink to training good data
            actual_train_good = data_path / "train" / self.config.dataset.normal_dir
            if actual_train_good.exists():
                try:
                    if os.name == 'nt':  # Windows
                        import shutil
                        shutil.copytree(actual_train_good, temp_good_link)
                        print(f"✅ Copied training data to {temp_good_link}")
                    else:  # Unix/Linux
                        temp_good_link.symlink_to(actual_train_good.absolute())
                        print(f"✅ Created symlink: {temp_good_link} -> {actual_train_good}")
                except Exception as symlink_error:
                    print(f"⚠️ Symlink failed: {symlink_error}")
                    # Fallback: copy directory
                    import shutil
                    shutil.copytree(actual_train_good, temp_good_link)
                    print(f"✅ Copied training data to {temp_good_link}")
            
            # For defective data, create empty directory or link to test defective
            test_defective = data_path / "test" / self.config.dataset.abnormal_dir
            if test_defective.exists() and len(list(test_defective.glob("*.[jp][pn]g"))) > 0:
                try:
                    if os.name == 'nt':  # Windows
                        import shutil
                        shutil.copytree(test_defective, temp_defective_link)
                        print(f"✅ Copied test defective data to {temp_defective_link}")
                    else:  # Unix/Linux
                        temp_defective_link.symlink_to(test_defective.absolute())
                        print(f"✅ Created symlink: {temp_defective_link} -> {test_defective}")
                except Exception as symlink_error:
                    print(f"⚠️ Defective symlink failed: {symlink_error}")
                    temp_defective_link.mkdir(exist_ok=True)
                    print(f"✅ Created empty defective directory: {temp_defective_link}")
            else:
                temp_defective_link.mkdir(exist_ok=True)
                print(f"✅ Created empty defective directory: {temp_defective_link}")
            
            # Now try with the new structure
            params['root'] = temp_train_root
            print(f"📦 Strategy 2 - Creating Folder datamodule with root: {temp_train_root}")
            
            try:
                self.datamodule = Folder(**params)
                print("✅ Strategy 2 successful - Data module initialized with reorganized structure")
                return
                
            except Exception as e2:
                print(f"❌ Strategy 2 failed: {e2}")
                
                # Strategy 3: Minimal fallback
                print(f"🔍 Trying Strategy 3: Minimal configuration")
                
                minimal_params = {
                    'root': temp_train_root,
                }
                
                if 'name' in folder_params:
                    minimal_params['name'] = self.config.dataset.name
                
                try:
                    self.datamodule = Folder(**minimal_params)
                    print("✅ Strategy 3 successful - Minimal data module initialized")
                    
                    # Manually set properties if possible
                    if hasattr(self.datamodule, 'train_batch_size'):
                        self.datamodule.train_batch_size = self.config.dataloader.train_batch_size
                    if hasattr(self.datamodule, 'eval_batch_size'):
                        self.datamodule.eval_batch_size = self.config.dataloader.eval_batch_size
                    if hasattr(self.datamodule, 'num_workers'):
                        self.datamodule.num_workers = self.config.dataloader.num_workers
                        
                except Exception as e3:
                    print(f"❌ All strategies failed!")
                    print(f"Strategy 1 error: {e1}")
                    print(f"Strategy 2 error: {e2}")
                    print(f"Strategy 3 error: {e3}")
                    raise RuntimeError("Failed to create datamodule with any strategy")
        
    def setup_engine(self):
        """Setup training engine with proper device detection"""
        print("⚙️ Setting up training engine...")
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(self.config.trainer.default_root_dir) / "checkpoints",
            filename="patchcore-{epoch:02d}",
            save_top_k=1,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Determine device and precision
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1
            # Use mixed precision for RTX 3080 Ti
            precision = "16-mixed" if torch.cuda.get_device_capability()[0] >= 7 else 32
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ Precision: {precision}")
        else:
            accelerator = "cpu"
            devices = 1
            precision = 32
            print("⚠️ Using CPU (training will be slower)")
        
        # Create engine
        self.engine = Engine(
            accelerator=accelerator,
            devices=devices,
            default_root_dir=self.config.trainer.default_root_dir,
            max_epochs=self.config.trainer.max_epochs,
            callbacks=callbacks,
            logger=True,
            log_every_n_steps=self.config.trainer.log_every_n_steps,
            val_check_interval=self.config.trainer.val_check_interval,
            precision=precision,
            enable_checkpointing=True,
            enable_progress_bar=True
        )
        
        print("✅ Training engine setup complete")
        
    def train(self):
        """Train the model with error recovery"""
        self.setup_engine()
        
        print("\n" + "="*50)
        print("🚀 Starting PatchCore training...")
        print("💡 Training on NORMAL data only (as expected for PatchCore)")
        print("="*50 + "\n")
        
        try:
            # Prepare datamodule if needed
            print("🔧 Setting up datamodule...")
            if hasattr(self.datamodule, 'setup'):
                self.datamodule.setup()
                print("✅ Datamodule setup completed")
            
            # Train the model
            print("🎯 Starting training process...")
            self.engine.fit(
                model=self.model,
                datamodule=self.datamodule
            )
            
            print("\n✅ Training completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Try with reduced settings
            print("\n🔄 Retrying with reduced settings...")
            
            # Reduce batch size and workers
            if hasattr(self.datamodule, 'train_batch_size'):
                old_batch = self.datamodule.train_batch_size
                self.datamodule.train_batch_size = max(1, old_batch // 2)
                print(f"📉 Reduced train batch size: {old_batch} -> {self.datamodule.train_batch_size}")
                
            if hasattr(self.datamodule, 'eval_batch_size'):
                old_eval = self.datamodule.eval_batch_size
                self.datamodule.eval_batch_size = max(1, old_eval // 2)
                print(f"📉 Reduced eval batch size: {old_eval} -> {self.datamodule.eval_batch_size}")
                
            if hasattr(self.datamodule, 'num_workers'):
                self.datamodule.num_workers = 0
                print(f"📉 Set num_workers to 0")
            
            # Recreate engine with reduced settings
            self.setup_engine()
            
            try:
                # Re-setup datamodule with new settings
                if hasattr(self.datamodule, 'setup'):
                    self.datamodule.setup()
                
                self.engine.fit(
                    model=self.model,
                    datamodule=self.datamodule
                )
                print("\n✅ Training completed with reduced settings")
                
            except Exception as e2:
                print(f"\n❌ Training failed even with reduced settings: {e2}")
                import traceback
                traceback.print_exc()
                raise
        
        # Save the model
        model_path = self._save_model()
        return model_path
        
    def _save_model(self) -> Path:
        """Save the trained model"""
        models_dir = Path(self.config.trainer.default_root_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "patchcore_model.pth"
        
        # Save model state and configuration
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': dict(self.config),
            'model_type': 'patchcore',
            'backbone': self.config.model.backbone,
            'image_size': list(self.config.dataset.image_size) if hasattr(self.config.dataset, 'image_size') else [256, 256],
            'normalization': self.config.dataset.normalization if hasattr(self.config.dataset, 'normalization') else 'imagenet'
        }, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        return model_path
        
    def evaluate(self) -> Dict:
        """Evaluate the trained model"""
        print("\n📊 Evaluating model performance...")
        
        try:
            # Test the model
            test_results = self.engine.test(
                model=self.model,
                datamodule=self.datamodule
            )
            
            if test_results and len(test_results) > 0:
                metrics = test_results[0]
                print("\n📈 Evaluation Results:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
                return metrics
            else:
                print("⚠️ No test results available")
                return {}
                
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            print("This is often due to missing test data or version incompatibilities")
            return {"error": str(e)}
        
    def prepare_for_deployment(self) -> Path:
        """Prepare model for Raspberry Pi deployment"""
        print("\n📦 Preparing model for deployment...")
        
        # Create deployment directory
        deploy_dir = Path("models/deployment")
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        deployment_model_path = deploy_dir / "patchcore_deployment.pth"
        
        # Create deployment package
        deployment_package = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'backbone': self.config.model.backbone,
                'layers': list(self.config.model.layers),
                'coreset_sampling_ratio': self.config.model.coreset_sampling_ratio,
                'num_neighbors': self.config.model.num_neighbors,
            },
            'image_size': list(self.config.dataset.image_size) if hasattr(self.config.dataset, 'image_size') else [256, 256],
            'normalization': self.config.dataset.normalization if hasattr(self.config.dataset, 'normalization') else 'imagenet',
            'tiling_config': {
                'tile_size': list(self.config.tiling.tile_size),
                'stride': list(self.config.tiling.stride),
                'apply': self.config.tiling.apply
            } if hasattr(self.config, 'tiling') else None,
        }
        
        torch.save(deployment_package, deployment_model_path)
        
        print(f"✅ Deployment package saved to: {deployment_model_path}")
        print(f"📦 Package size: {deployment_model_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return deployment_model_path