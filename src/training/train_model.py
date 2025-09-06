# src/training/train_model.py
import os
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore
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
        self.model = Patchcore(
            backbone=self.config.model.backbone,
            layers=self.config.model.layers,
            pre_trained=self.config.model.pre_trained,
            coreset_sampling_ratio=self.config.model.coreset_sampling_ratio,
            num_neighbors=self.config.model.num_neighbors,
        )
        
    def setup_data(self):
        """Setup data module"""
        print("Setting up data module...")
        
        # Ensure data directories exist
        data_path = Path(self.config.dataset.path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
            
        self.datamodule = Folder(
            name=self.config.dataset.name,
            root=data_path,
            normal_dir=self.config.dataset.normal_dir,
            abnormal_dir=self.config.dataset.abnormal_dir,
            task=self.config.dataset.task,
            image_size=tuple(self.config.dataset.image_size),
            center_crop=self.config.dataset.center_crop,
            normalization=self.config.dataset.normalization,
            train_batch_size=self.config.dataloader.train_batch_size,
            eval_batch_size=self.config.dataloader.eval_batch_size,
            num_workers=self.config.dataloader.num_workers,
        )
        
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
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping (though PatchCore typically needs only 1 epoch)
        early_stopping = EarlyStopping(
            monitor="image_AUROC",
            patience=5,
            mode="max",
        )
        callbacks.append(early_stopping)
        
        self.engine = Engine(
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            default_root_dir=self.config.trainer.default_root_dir,
            max_epochs=self.config.trainer.max_epochs,
            callbacks=callbacks,
            logger=True,
            log_every_n_steps=self.config.trainer.log_every_n_steps,
            val_check_interval=self.config.trainer.val_check_interval,
            precision=self.config.trainer.precision if hasattr(self.config.trainer, 'precision') else 32,
        )
        
    def train(self):
        """Train the model"""
        self.setup_engine()
        
        print("Starting training process...")
        self.engine.fit(
            model=self.model,
            datamodule=self.datamodule,
        )
        
        # Get the best model path
        model_path = Path(self.config.trainer.default_root_dir) / "patchcore_model.pth"
        
        # Save the model in a format suitable for deployment
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_type': 'patchcore'
        }, model_path)
        
        return model_path
        
    def evaluate(self):
        """Evaluate the trained model"""
        print("Evaluating model performance...")
        
        # Test the model
        test_results = self.engine.test(
            model=self.model,
            datamodule=self.datamodule,
        )
        
        return test_results[0] if test_results else {}
        
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
        
        # Save configuration for Raspberry Pi
        import yaml
        config_path = deploy_dir / "deployment_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(deployment_package, f, default_flow_style=False)
            
        print(f"Deployment package saved to: {deployment_model_path}")
        
        return deployment_model_path

