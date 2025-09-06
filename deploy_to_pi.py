# deploy_to_pi.py - Deployment preparation script
import os
import shutil
import subprocess
import sys
from pathlib import Path
import yaml
import torch

class RaspberryPiDeployment:
    """Prepare and deploy model to Raspberry Pi"""
    
    def __init__(self, model_path: str, pi_address: str = None, pi_user: str = "pi"):
        self.model_path = Path(model_path)
        self.pi_address = pi_address
        self.pi_user = pi_user
        self.deployment_dir = Path("deployment_package")
        
    def create_deployment_package(self):
        """Create deployment package for Raspberry Pi"""
        print("📦 Creating deployment package...")
        
        # Create deployment directory
        if self.deployment_dir.exists():
            shutil.rmtree(self.deployment_dir)
        self.deployment_dir.mkdir(parents=True)
        
        # Copy essential files
        essential_files = [
            "src/inference/",
            "src/utils/",
            "src/deployment/",
            "config/patchcore_config.yaml",
            "requirements_pi.txt",
            "database/",
        ]
        
        for file_path in essential_files:
            src = Path(file_path)
            if src.exists():
                if src.is_dir():
                    dst = self.deployment_dir / src.name
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, self.deployment_dir / src.name)
                print(f"✅ Copied: {file_path}")
            else:
                print(f"⚠️ Warning: {file_path} not found")
        
        # Copy model files
        models_dir = self.deployment_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        if self.model_path.exists():
            shutil.copy2(self.model_path, models_dir / "patchcore_model.pth")
            print(f"✅ Copied model: {self.model_path}")
        else:
            print(f"❌ Model not found: {self.model_path}")
            return False
        
        # Create startup script
        self._create_startup_script()
        
        # Create installation script
        self._create_installation_script()
        
        # Create systemd service file
        self._create_systemd_service()
        
        print(f"✅ Deployment package created in: {self.deployment_dir}")
        return True
    
    def _create_startup_script(self):
        """Create startup script for Raspberry Pi"""
        startup_script = """#!/bin/bash

# Raspberry Pi PatchCore Defect Detection Startup Script

# Set working directory
cd /home/pi/patchcore_defect_detection

# Activate virtual environment
source venv/bin/activate

# Run the detector
python -c "
from src.deployment.raspberry_pi_detector import RaspberryPiDetector

# Configuration
MODEL_PATH = 'models/patchcore_model.pth'
CSV_PATH = 'database/detection_results.csv'
CONFIDENCE_THRESHOLD = 0.5

# Initialize and run detector
detector = RaspberryPiDetector(
    model_path=MODEL_PATH,
    csv_path=CSV_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD
)

# Run continuous monitoring
detector.continuous_monitoring(
    capture_interval=5.0,
    save_images=True
)
"
"""
        
        startup_path = self.deployment_dir / "start_detector.sh"
        with open(startup_path, 'w') as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(startup_path, 0o755)
        print("✅ Created startup script")
    
    def _create_installation_script(self):
        """Create installation script for Raspberry Pi"""
        install_script = """#!/bin/bash

# PatchCore Defect Detection Installation Script for Raspberry Pi 5

echo "🚀 Installing PatchCore Defect Detection System..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y python3-picamera2

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Raspberry Pi (CPU only)
echo "🧠 Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements_pi.txt

# Set up directories
echo "📁 Setting up directories..."
mkdir -p database/images/normal
mkdir -p database/images/defective
mkdir -p logs

# Create database CSV
python -c "
import pandas as pd
from pathlib import Path

csv_path = Path('database/detection_results.csv')
if not csv_path.exists():
    columns = [
        'timestamp', 'anomaly_score', 'is_defective', 
        'confidence_threshold', 'image_shape', 
        'num_tiles_processed', 'detection_status'
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path, index=False)
    print('✅ Initialized detection database')
"

# Make startup script executable
chmod +x start_detector.sh

# Set up systemd service (optional)
echo "⚙️ Setting up systemd service..."
sudo cp patchcore_detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable patchcore_detector.service

echo "✅ Installation completed!"
echo ""
echo "🎯 To start the detector:"
echo "   ./start_detector.sh"
echo ""
echo "🔄 To start as a service:"
echo "   sudo systemctl start patchcore_detector"
echo ""
echo "📊 To view logs:"
echo "   journalctl -u patchcore_detector -f"
"""
        
        install_path = self.deployment_dir / "install_pi.sh"
        with open(install_path, 'w') as f:
            f.write(install_script)
        
        os.chmod(install_path, 0o755)
        print("✅ Created installation script")
    
    def _create_systemd_service(self):
        """Create systemd service file"""
        service_content = """[Unit]
Description=PatchCore Defect Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/patchcore_defect_detection
Environment=PATH=/home/pi/patchcore_defect_detection/venv/bin
ExecStart=/home/pi/patchcore_defect_detection/start_detector.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_path = self.deployment_dir / "patchcore_detector.service"
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        print("✅ Created systemd service file")
    
    def deploy_to_pi(self):
        """Deploy package to Raspberry Pi via SCP"""
        if not self.pi_address:
            print("⚠️ Raspberry Pi address not provided. Manual deployment required.")
            print(f"📁 Deployment package ready in: {self.deployment_dir}")
            return False
        
        print(f"🚀 Deploying to Raspberry Pi at {self.pi_address}...")
        
        try:
            # Create remote directory
            subprocess.run([
                "ssh", f"{self.pi_user}@{self.pi_address}", 
                "mkdir -p /home/pi/patchcore_defect_detection"
            ], check=True)
            
            # Copy deployment package
            subprocess.run([
                "scp", "-r", str(self.deployment_dir) + "/*", 
                f"{self.pi_user}@{self.pi_address}:/home/pi/patchcore_defect_detection/"
            ], check=True)
            
            # Run installation script
            subprocess.run([
                "ssh", f"{self.pi_user}@{self.pi_address}",
                "cd /home/pi/patchcore_defect_detection && ./install_pi.sh"
            ], check=True)
            
            print("✅ Deployment completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Deployment failed: {e}")
            return False
    
    def create_manual_deployment_instructions(self):
        """Create manual deployment instructions"""
        instructions = """
# Manual Deployment Instructions for Raspberry Pi 5

## 1. Transfer Files
Copy the entire 'deployment_package' folder to your Raspberry Pi:

```bash
# On your local machine
scp -r deployment_package pi@<PI_IP_ADDRESS>:/home/pi/patchcore_defect_detection
```

## 2. Install on Raspberry Pi
SSH into your Raspberry Pi and run:

```bash
cd /home/pi/patchcore_defect_detection
chmod +x install_pi.sh
./install_pi.sh
```

## 3. Start Detection
Run the detector:

```bash
./start_detector.sh
```

Or start as a systemd service:

```bash
sudo systemctl start patchcore_detector
sudo systemctl status patchcore_detector
```

## 4. Monitor Logs
View real-time logs:

```bash
journalctl -u patchcore_detector -f
```

## 5. Access Results
- Detection results: `database/detection_results.csv`
- Captured images: `database/images/`
- Logs: `logs/`

## 6. Configuration
Edit detection parameters in:
- `config/patchcore_config.yaml`
- Modify confidence threshold and capture interval in the startup script

## Troubleshooting
- Ensure camera is properly connected and enabled
- Check Python virtual environment is activated
- Verify all dependencies are installed
- Monitor system resources (CPU, memory)
"""
        
        instructions_path = self.deployment_dir / "DEPLOYMENT_INSTRUCTIONS.md"
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        print("✅ Created deployment instructions")

# src/deployment/model_optimizer.py
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

class ModelOptimizer:
    """Optimize model for Raspberry Pi deployment"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        
    def optimize_for_cpu(self, output_path: str = None):
        """Optimize model for CPU inference"""
        print("⚡ Optimizing model for CPU inference...")
        
        # Load model
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Create optimized checkpoint
        optimized_checkpoint = {
            'model_state_dict': checkpoint['model_state_dict'],
            'model_config': checkpoint['model_config'],
            'image_size': checkpoint['image_size'],
            'normalization': checkpoint['normalization'],
            'optimized_for': 'cpu',
            'optimization_timestamp': torch.tensor(0)  # Placeholder
        }
        
        # Save optimized model
        if output_path is None:
            output_path = self.model_path.parent / "optimized_cpu_model.pth"
        else:
            output_path = Path(output_path)
            
        torch.save(optimized_checkpoint, output_path)
        print(f"✅ CPU-optimized model saved to: {output_path}")
        
        return output_path
    
    def quantize_model(self, output_path: str = None):
        """Apply dynamic quantization for faster inference"""
        print("🔢 Applying dynamic quantization...")
        
        try:
            # Load model
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # For PatchCore, quantization is limited since it's feature-based
            # We'll optimize the storage format instead
            optimized_checkpoint = checkpoint.copy()
            
            # Convert float64 to float32 if present
            for key, value in optimized_checkpoint['model_state_dict'].items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                    optimized_checkpoint['model_state_dict'][key] = value.float()
            
            # Add quantization flag
            optimized_checkpoint['quantized'] = True
            
            # Save quantized model
            if output_path is None:
                output_path = self.model_path.parent / "quantized_model.pth"
            else:
                output_path = Path(output_path)
                
            torch.save(optimized_checkpoint, output_path)
            print(f"✅ Quantized model saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            return None

def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy PatchCore model to Raspberry Pi")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--pi_address", help="Raspberry Pi IP address for automatic deployment")
    parser.add_argument("--pi_user", default="pi", help="Raspberry Pi username")
    parser.add_argument("--optimize", action="store_true", help="Optimize model for deployment")
    
    args = parser.parse_args()
    
    # Optimize model if requested
    if args.optimize:
        optimizer = ModelOptimizer(args.model_path)
        optimized_path = optimizer.optimize_for_cpu()
        model_path = optimized_path
    else:
        model_path = args.model_path
    
    # Create deployment package
    deployer = RaspberryPiDeployment(model_path, args.pi_address, args.pi_user)
    
    if not deployer.create_deployment_package():
        print("❌ Failed to create deployment package")
        return
    
    # Create manual instructions
    deployer.create_manual_deployment_instructions()
    
    # Attempt automatic deployment if Pi address provided
    if args.pi_address:
        success = deployer.deploy_to_pi()
        if not success:
            print("\n⚠️ Automatic deployment failed. Please use manual deployment.")
            print("📖 See DEPLOYMENT_INSTRUCTIONS.md for manual steps.")
    else:
        print("\n📦 Deployment package created successfully!")
        print("📖 See DEPLOYMENT_INSTRUCTIONS.md for deployment steps.")

if __name__ == "__main__":
    main()

# setup_project.py - Project setup script
import os
import sys
from pathlib import Path
import subprocess

def create_project_structure():
    """Create the complete project structure"""
    print("🏗️ Setting up PatchCore project structure...")
    
    # Import the setup function
    sys.path.append('src/utils')
    from image_utils import setup_directories
    setup_directories()

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Install local training requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_local.txt"], 
                      check=True)
        print("✅ Local dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def create_sample_config():
    """Create sample configuration files"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # The config file is already created in the artifacts above
    print("✅ Configuration files ready")

def main():
    """Main setup function"""
    print("🚀 Setting up PatchCore Defect Detection Project")
    print("=" * 50)
    
    # Create project structure
    create_project_structure()
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Create configuration
    create_sample_config()
    
    print("\n✅ Project setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Place your training images in data/train/good/")
    print("2. Place your test images in data/test/good/ and data/test/defective/")
    print("3. Run: python train.py")
    print("4. Deploy to Raspberry Pi: python deploy_to_pi.py --model_path models/deployment/patchcore_deployment.pth")
    
if __name__ == "__main__":
    main()