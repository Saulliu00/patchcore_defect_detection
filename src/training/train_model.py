# src/training/train_model.py - Fixed version with direct imports
import os
from pathlib import Path
import torch

# Use direct imports to avoid problematic modules
try:
    # Try specific PatchCore import first
    from anomalib.models.image.patchcore import Patchcore
    print("✅ Using specific PatchCore import")
except ImportError:
    try:
        # Fallback to general import
        from anomalib.models import Patchcore
        print("✅ Using general PatchCore import")
    except ImportError as e:
        print(f"❌ Failed to import PatchCore: {e}")
        print("Try: pip uninstall anomalib && pip install anomalib==1.0.1")
        raise

from anomalib.data import Folder
from anomalib.engine import Engine
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

class PatchCoreTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.datamodule = None
        self.engine = None
        self.setup_model()
        self.setup_data()
        
    def setup_model(self):
        """Initialize PatchCore model"""
        print("Initializing PatchCore model...")
        
        # Create model with explicit parameters
        try:
            self.model = Patchcore(
                backbone=self.config.model.backbone,
                layers=self.config.model.layers,
                pre_trained=self.config.model.pre_trained,
                coreset_sampling_ratio=self.config.model.coreset_sampling_ratio,
                num_neighbors=self.config.model.num_neighbors)
            print("✅ PatchCore model initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize PatchCore: {e}")
            raise
        
    def setup_data(self):
        """Setup data module"""
        print("Setting up data module...")
        
        # Ensure data directories exist
        data_path = Path(self.config.dataset.path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
            
        # Validate training data exists
        train_good_path = data_path / "train" / self.config.dataset.normal_dir
        if not train_good_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_good_path}")
            
        # Count training images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        training_images = []
        for ext in image_extensions:
            training_images.extend(list(train_good_path.glob(ext)))
            
        if len(training_images) == 0:
            raise ValueError(f"No training images found in {train_good_path}")
            
        print(f"Found {len(training_images)} training images")
            
        try:
            # Create Folder datamodule with correct parameters for Anomalib 1.0.1
            self.datamodule = Folder(
                root=data_path,
                normal_dir=self.config.dataset.normal_dir,
                abnormal_dir=self.config.dataset.abnormal_dir,
                train_batch_size=self.config.dataloader.train_batch_size,
                eval_batch_size=self.config.dataloader.eval_batch_size,
                num_workers=self.config.dataloader.num_workers)
            print("✅ Data module setup successfully")
        except Exception as e:
            print(f"❌ Failed to setup data module: {e}")
            print("Trying with minimal parameters...")
            
            # Fallback with minimal parameters
            try:
                self.datamodule = Folder(
                    root=data_path,
                    normal_dir=self.config.dataset.normal_dir,
                    abnormal_dir=self.config.dataset.abnormal_dir,
                    train_batch_size=self.config.dataloader.train_batch_size,
                    eval_batch_size=self.config.dataloader.eval_batch_size)
                print("✅ Data module setup successfully with minimal parameters")
            except Exception as e2:
                print(f"❌ Even minimal setup failed: {e2}")
                raise
        
    def setup_engine(self):
        """Setup training engine"""
        print("Setting up training engine...")
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.trainer.default_root_dir,
            filename="patchcore-{epoch:02d}",
            save_top_k=1,
            monitor="image_AUROC",
            mode="max",
            save_last=True)
        callbacks.append(checkpoint_callback)
        
        # Get precision setting
        precision = getattr(self.config.trainer, 'precision', 32)
        if precision == 16 and not torch.cuda.is_available():
            print("WARNING: Mixed precision requested but CUDA not available, using 32-bit")
            precision = 32
        
        try:
            self.engine = Engine(
                accelerator=self.config.trainer.accelerator,
                devices=self.config.trainer.devices,
                default_root_dir=self.config.trainer.default_root_dir,
                max_epochs=self.config.trainer.max_epochs,
                callbacks=callbacks,
                logger=True,
                log_every_n_steps=self.config.trainer.log_every_n_steps,
                val_check_interval=self.config.trainer.val_check_interval,
                precision=precision)
            print("✅ Training engine setup successfully")
        except Exception as e:
            print(f"❌ Failed to setup training engine: {e}")
            raise
        
    def train(self):
        """Train the model"""
        self.setup_engine()
        
        print("Starting training process...")
        try:
            self.engine.fit(model=self.model,
                datamodule=self.datamodule)
            print("✅ Training completed successfully")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("Trying with reduced settings...")
            
            # Fallback: reduce batch size and workers
            self.config.dataloader.train_batch_size = max(1, self.config.dataloader.train_batch_size // 2)
            self.config.dataloader.eval_batch_size = max(1, self.config.dataloader.eval_batch_size // 2)
            self.config.dataloader.num_workers = min(2, self.config.dataloader.num_workers)
            
            print(f"Retrying with batch_size={self.config.dataloader.train_batch_size}, workers={self.config.dataloader.num_workers}")
            
            # Recreate datamodule with reduced settings
            self.setup_data()
            self.setup_engine()
            
            try:
                self.engine.fit(model=self.model,
                    datamodule=self.datamodule)
                print("✅ Training completed with reduced settings")
            except Exception as e2:
                print(f"❌ Training failed even with reduced settings: {e2}")
                raise
        
        # Get the best model path
        model_path = Path(self.config.trainer.default_root_dir) / "patchcore_model.pth"
        
        # Save the model in a format suitable for deployment
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_type': 'patchcore'
        }, model_path)
        
        print(f"Model saved to: {model_path}")
        return model_path
        
    def evaluate(self):
        """Evaluate the trained model"""
        print("Evaluating model performance...")
        
        try:
            # Test the model
            test_results = self.engine.test(
                model=self.model,
                datamodule=self.datamodule)
            
            print("✅ Evaluation completed successfully")
            return test_results[0] if test_results else {}
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {"error": str(e)}
        
    def prepare_for_deployment(self):
        """Prepare model for Raspberry Pi deployment"""
        print("Preparing model for deployment...")
        
        # Create deployment directory
        deploy_dir = Path("models/deployment")
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model in deployment format
        deployment_model_path = deploy_dir / "patchcore_deployment.pth"
        
        # Save model with minimal dependencies
        deployment_package = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'backbone': self.config.model.backbone,
                'layers': self.config.model.layers,
                'coreset_sampling_ratio': self.config.model.coreset_sampling_ratio,
                'num_neighbors': self.config.model.num_neighbors,
            },
            'image_size': self.config.dataset.image_size,
            'normalization': self.config.dataset.normalization,
            'tiling_config': self.config.tiling if hasattr(self.config, 'tiling') else None,
        }
        
        torch.save(deployment_package, deployment_model_path)
        
        print(f"✅ Deployment package saved to: {deployment_model_path}")
        return deployment_model_path# src/training/train_model.py - Fixed version with safer imports
import os
from pathlib import Path
import torch

# Use more specific imports to avoid VLM dependencies
try:
    from anomalib.data import Folder
    from anomalib.models.image.patchcore import Patchcore
    from anomalib.engine import Engine
except ImportError:
    # Fallback for older Anomalib versions
    from anomalib.data import Folder
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

# --- Windows-safe patch for anomalib versioned workspace ---
try:
    from anomalib.utils import path as _anom_path
    from pathlib import Path as _Path

    def _safe_create_versioned_dir(root_dir):
        root_dir = _Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        # find next vN
        n = 0
        while (root_dir / f"v{n}").exists():
            n += 1
        new_version_dir = root_dir / f"v{n}"
        new_version_dir.mkdir(parents=True, exist_ok=True)
        # Try to create 'latest' symlink; fall back silently if not permitted (Windows without dev mode/admin)
        latest_link_path = root_dir / "latest"
        try:
            if latest_link_path.exists() or latest_link_path.is_symlink():
                try:
                    latest_link_path.unlink()
                except Exception:
                    pass
            latest_link_path.symlink_to(new_version_dir, target_is_directory=True)
        except Exception:
            # Best effort: ensure a normal folder exists so downstream code that writes into 'latest' won't explode
            try:
                latest_link_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return new_version_dir

    # Monkeypatch anomalib to use the safe function
    _anom_path.create_versioned_dir = _safe_create_versioned_dir
except Exception:
    # If anomalib import changes in future versions, fail open and let default behavior run
    pass
# --- end Windows-safe patch ---

class PatchCoreTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.datamodule = None
        self.engine = None
        self.setup_model()
        self.setup_data()
        
    def setup_model(self):
        """Initialize PatchCore model"""
        print("Initializing PatchCore model...")
        
        # Create model with explicit parameters
        self.model = Patchcore(
            backbone=self.config.model.backbone,
            layers=self.config.model.layers,
            pre_trained=self.config.model.pre_trained,
            coreset_sampling_ratio=self.config.model.coreset_sampling_ratio,
            num_neighbors=self.config.model.num_neighbors)
        
    def setup_data(self):
        """Setup data module"""
        print("Setting up data module...")
        
        # Ensure data directories exist
        data_path = Path(self.config.dataset.path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
            
        # Validate training data exists
        train_good_path = data_path / "train" / self.config.dataset.normal_dir
        if not train_good_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_good_path}")
            
        # Count training images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        training_images = []
        for ext in image_extensions:
            training_images.extend(list(train_good_path.glob(ext)))
            
        if len(training_images) == 0:
            raise ValueError(f"No training images found in {train_good_path}")
            
        print(f"Found {len(training_images)} training images")
            
        self.datamodule = Folder(
            name=self.config.dataset.name,
            root=data_path,
            normal_dir=self.config.dataset.normal_dir,
            abnormal_dir=self.config.dataset.abnormal_dir,
            train_batch_size=self.config.dataloader.train_batch_size,
            eval_batch_size=self.config.dataloader.eval_batch_size,
            num_workers=self.config.dataloader.num_workers)
        
    def setup_engine(self):
        """Setup training engine"""
        print("Setting up training engine...")
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.trainer.default_root_dir,
            filename="patchcore-{epoch:02d}",
            save_top_k=1,
            monitor="image_AUROC",
            mode="max",
            save_last=True)
        callbacks.append(checkpoint_callback)
        
        # Get precision setting
        precision = getattr(self.config.trainer, 'precision', 32)
        if precision == 16 and not torch.cuda.is_available():
            print("WARNING: Mixed precision requested but CUDA not available, using 32-bit")
            precision = 32
        
        self.engine = Engine(
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            default_root_dir=self.config.trainer.default_root_dir,
            max_epochs=self.config.trainer.max_epochs,
            callbacks=callbacks,
            logger=True,
            log_every_n_steps=self.config.trainer.log_every_n_steps,
            val_check_interval=self.config.trainer.val_check_interval,
            precision=precision)
        
    def train(self):
        """Train the model"""
        self.setup_engine()
        
        print("Starting training process...")
        try:
            self.engine.fit(model=self.model,
                datamodule=self.datamodule)
        except Exception as e:
            print(f"Training failed: {e}")
            print("Trying with reduced settings...")
            
            # Fallback: reduce batch size and workers
            self.config.dataloader.train_batch_size = max(1, self.config.dataloader.train_batch_size // 2)
            self.config.dataloader.eval_batch_size = max(1, self.config.dataloader.eval_batch_size // 2)
            self.config.dataloader.num_workers = min(2, self.config.dataloader.num_workers)
            
            # Recreate datamodule with reduced settings
            self.setup_data()
            self.setup_engine()
            
            self.engine.fit(model=self.model,
                datamodule=self.datamodule)
        
        # Get the best model path
        model_path = Path(self.config.trainer.default_root_dir) / "patchcore_model.pth"
        
        # Save the model in a format suitable for deployment
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_type': 'patchcore'
        }, model_path)
        
        print(f"Model saved to: {model_path}")
        return model_path
        
    def evaluate(self):
        """Evaluate the trained model"""
        print("Evaluating model performance...")
        
        try:
            # Test the model
            test_results = self.engine.test(
                model=self.model,
                datamodule=self.datamodule)
            
            return test_results[0] if test_results else {}
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {"error": str(e)}
        
    def prepare_for_deployment(self):
        """Prepare model for Raspberry Pi deployment"""
        print("Preparing model for deployment...")
        
        # Create deployment directory
        deploy_dir = Path("models/deployment")
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model in deployment format
        deployment_model_path = deploy_dir / "patchcore_deployment.pth"
        
        # Save model with minimal dependencies
        deployment_package = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'backbone': self.config.model.backbone,
                'layers': self.config.model.layers,
                'coreset_sampling_ratio': self.config.model.coreset_sampling_ratio,
                'num_neighbors': self.config.model.num_neighbors,
            },
            'image_size': self.config.dataset.image_size,
            'normalization': self.config.dataset.normalization,
            'tiling_config': self.config.tiling if hasattr(self.config, 'tiling') else None,
        }
        
        torch.save(deployment_package, deployment_model_path)
        
        print(f"Deployment package saved to: {deployment_model_path}")
        return deployment_model_path