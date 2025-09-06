# src/training/train_model.py - Fixed and verified version
import os
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# Import with compatibility handling
try:
    # Try Anomalib 1.0+ imports
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

class PatchCoreTrainer:
    """PatchCore model trainer with robust error handling"""
    
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
        """Setup data module with validation"""
        print("📊 Setting up data module...")
        
        # Validate data directory
        data_path = Path(self.config.dataset.path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
            
        # Check for training data
        train_good_path = data_path / "train" / self.config.dataset.normal_dir
        if not train_good_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_good_path}")
            
        # Count training images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        training_images = []
        for ext in image_extensions:
            training_images.extend(list(train_good_path.glob(ext)))
            training_images.extend(list(train_good_path.glob(ext.upper())))
            
        if len(training_images) == 0:
            raise ValueError(f"No training images found in {train_good_path}")
            
        print(f"✅ Found {len(training_images)} training images")
        
        # Validate test data
        test_good_path = data_path / "test" / self.config.dataset.normal_dir
        test_defective_path = data_path / "test" / self.config.dataset.abnormal_dir
        
        test_good_count = len(list(test_good_path.glob("*.[jp][pn]g"))) if test_good_path.exists() else 0
        test_defective_count = len(list(test_defective_path.glob("*.[jp][pn]g"))) if test_defective_path.exists() else 0
        
        print(f"✅ Found {test_good_count} normal test images")
        print(f"✅ Found {test_defective_count} defective test images")
        
        if test_good_count == 0:
            print("⚠️ Warning: No normal test images found - evaluation will be limited")
        if test_defective_count == 0:
            print("⚠️ Warning: No defective test images found - evaluation will be limited")
            
        # Setup Folder datamodule with proper parameters
        try:
            # Try with all parameters first
            self.datamodule = Folder(
                name=self.config.dataset.name,
                root=data_path,
                normal_dir=self.config.dataset.normal_dir,
                abnormal_dir=self.config.dataset.abnormal_dir,
                normal_split_ratio=0.8,  # 80% for training
                seed=42,
                train_batch_size=self.config.dataloader.train_batch_size,
                eval_batch_size=self.config.dataloader.eval_batch_size,
                num_workers=self.config.dataloader.num_workers,
                task=self.config.dataset.task,
                image_size=self.config.dataset.image_size
            )
            print("✅ Data module initialized with full configuration")
            
        except TypeError as e:
            # Fallback to minimal parameters
            print(f"⚠️ Using minimal datamodule parameters: {e}")
            self.datamodule = Folder(
                root=data_path,
                normal_dir=self.config.dataset.normal_dir,
                abnormal_dir=self.config.dataset.abnormal_dir,
                train_batch_size=self.config.dataloader.train_batch_size,
                eval_batch_size=self.config.dataloader.eval_batch_size,
                num_workers=self.config.dataloader.num_workers
            )
            print("✅ Data module initialized with minimal configuration")
        
    def setup_engine(self):
        """Setup training engine with proper device detection"""
        print("⚙️ Setting up training engine...")
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(self.config.trainer.default_root_dir) / "checkpoints",
            filename="patchcore-{epoch:02d}-{image_AUROC:.3f}",
            save_top_k=1,
            monitor="image_AUROC",
            mode="max",
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
        print("="*50 + "\n")
        
        try:
            # Prepare datamodule
            self.datamodule.setup()
            
            # Train the model
            self.engine.fit(
                model=self.model,
                datamodule=self.datamodule
            )
            
            print("\n✅ Training completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            
            # Try with reduced settings
            print("\n🔄 Retrying with reduced settings...")
            
            # Reduce batch size and workers
            self.config.dataloader.train_batch_size = max(1, self.config.dataloader.train_batch_size // 2)
            self.config.dataloader.eval_batch_size = max(1, self.config.dataloader.eval_batch_size // 2)
            self.config.dataloader.num_workers = 0  # Disable multiprocessing
            
            print(f"📉 Reduced batch_size to {self.config.dataloader.train_batch_size}")
            print(f"📉 Disabled multiprocessing")
            
            # Recreate components with reduced settings
            self.setup_data()
            self.setup_engine()
            
            try:
                self.datamodule.setup()
                self.engine.fit(
                    model=self.model,
                    datamodule=self.datamodule
                )
                print("\n✅ Training completed with reduced settings")
                
            except Exception as e2:
                print(f"\n❌ Training failed even with reduced settings: {e2}")
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
            'image_size': self.config.dataset.image_size,
            'normalization': self.config.dataset.normalization
        }, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        return model_path
        
    def evaluate(self) -> Dict:
        """Evaluate the trained model"""
        print("\n📊 Evaluating model performance...")
        
        try:
            # Ensure datamodule is setup
            if not hasattr(self.datamodule, 'test_dataloader'):
                self.datamodule.setup()
            
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
            'image_size': list(self.config.dataset.image_size),
            'normalization': self.config.dataset.normalization,
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