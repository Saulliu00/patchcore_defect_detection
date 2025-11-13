# PatchCore Defect Detection System

A complete solution for manufacturing defect detection using PatchCore anomaly detection, optimized for NVIDIA RTX 3080 Ti training and Raspberry Pi 5 deployment.

## Features

- **PatchCore Anomaly Detection**: State-of-the-art anomaly detection using Anomalib 2.x
- **High-Resolution Support**: Works with 512x512 images (configurable)
- **Raspberry Pi Ready**: Standalone inference without Anomalib dependency
- **Visualization Tools**: Detailed per-image results with anomaly heatmaps
- **Simple Workflow**: Minimal scripts for training, validation, and deployment

## Quick Start

### Prerequisites
- Python 3.12+ (tested with Python 3.12.10)
- NVIDIA GPU with CUDA support (tested on RTX 3080 Ti)
- For deployment: Raspberry Pi 5

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Saulliu00/patchcore_defect_detection.git
cd patchcore_defect_detection
```

2. **Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Environment Details:**
- Python: 3.12.10
- PyTorch: 2.9.0
- Anomalib: 2.2.0
- Lightning: 2.5.6

### Data Preparation

Organize your images in this structure:
```
data/
├── train/
│   └── good/              # Normal parts only (200+ images recommended)
├── test/
│   ├── good/             # Normal test images (50+ images)
│   └── defective/        # Defective test images (20+ images)
```

**Image requirements:**
- Resolution: 512x512 recommended (configurable in `config/patchcore_config.yaml`)
- Format: JPG or PNG
- Consistent lighting and positioning
- Clear, focused images

## Training

### Train the Model

```bash
python train.py
```

**What happens:**
- Trains PatchCore model with ResNet18 backbone
- Uses only normal (good) images from `data/train/good/`
- Saves model to `models/minimal/patchcore_minimal.pth`
- Training is fast (1 epoch) as PatchCore learns from features

**Configuration:**
- Edit `config/patchcore_config.yaml` to adjust:
  - Image size (default: 512x512)
  - Backbone (default: resnet18)
  - Coreset sampling ratio (default: 0.1)
  - Batch size (default: 1)

## Validation

### Basic Validation

```bash
python validate_model.py
```

**Outputs:**
- Overall accuracy and metrics
- Mean anomaly scores for normal vs defective images
- Optimal threshold recommendation

### Validation with Visualization

```bash
python validate_with_visualization.py
```

**Outputs:**
- `validation_results/detailed_results.csv` - Per-image scores and predictions
- `validation_results/visualizations/` - Heatmap overlays for each image
- Detailed metrics including precision, recall, F1 score
- List of misclassified images

**CSV Format:**
```csv
filename,category,true_label,predicted_label,score,correct,path
image001.jpg,normal,normal,normal,0.8234,True,data/test/good/image001.jpg
```

## Deployment to Raspberry Pi

### Step 1: Generate Deployment Package

```bash
python deploy_to_pi.py
```

**What it creates in `pi_deployment/`:**
- `pi_inference.py` - Standalone inference script (no Anomalib needed)
- `patchcore_pi.pth` - Optimized model checkpoint
- `requirements.txt` - Minimal dependencies (torch, torchvision, opencv, pillow)
- `README.md` - Deployment instructions

### Step 2: Transfer to Raspberry Pi

```bash
# Copy the entire pi_deployment folder to your Raspberry Pi
scp -r pi_deployment/ pi@raspberrypi.local:~/
```

### Step 3: Run Inference on Raspberry Pi

On the Raspberry Pi:

```bash
cd ~/pi_deployment

# Install dependencies
pip3 install -r requirements.txt

# Run inference on images
python3 pi_inference.py --input /path/to/images --output results.csv

# Optional arguments:
# --threshold 0.95       # Anomaly threshold (default: 0.95)
# --checkpoint path.pth  # Model checkpoint (default: patchcore_pi.pth)
# --device cpu           # Device: cpu or cuda (default: cpu)
```

**Output CSV Format:**
```csv
filename,score,predicted_label,path
image001.jpg,0.8234,normal,/path/to/images/image001.jpg
image002.jpg,0.9876,defective,/path/to/images/image002.jpg
```

## Project Structure

```
patchcore_defect_detection/
├── config/
│   └── patchcore_config.yaml      # Configuration (image size, model params)
├── data/
│   ├── train/good/                # Training images (normal only)
│   ├── test/good/                 # Test normal images
│   └── test/defective/            # Test defective images
├── models/
│   └── minimal/
│       └── patchcore_minimal.pth  # Trained model
├── pi_deployment/                  # Generated deployment package
│   ├── pi_inference.py            # Standalone inference script
│   ├── patchcore_pi.pth           # Optimized checkpoint
│   ├── requirements.txt           # Minimal dependencies
│   └── README.md                  # Deployment instructions
├── validation_results/             # Validation outputs
│   ├── detailed_results.csv       # Per-image results
│   └── visualizations/            # Heatmap images
├── train.py                       # Training script
├── validate_model.py              # Basic validation
├── validate_with_visualization.py # Detailed validation with viz
└── deploy_to_pi.py                # Generate deployment package
```

## Hardware Requirements

### Training Machine
- **GPU**: NVIDIA RTX 3080 Ti (12GB VRAM) or equivalent
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ available space
- **OS**: Linux/macOS/Windows with CUDA 11.8+

### Raspberry Pi 5 Deployment
- Raspberry Pi 5 (4GB/8GB recommended)
- Camera module or USB camera (optional)
- MicroSD card (32GB+ Class 10)
- Adequate power supply (27W USB-C)

## Model Architecture

**PatchCore Details:**
- **Backbone**: ResNet18 (lightweight, fast inference)
- **Layers**: layer2, layer3 feature extraction
- **Training**: One-class learning (normal samples only)
- **Memory Bank**: Stores representative feature patches
- **Scoring**: K-nearest neighbors distance (k=9)
- **Coreset Sampling**: 10% of features for efficiency

## Configuration Reference

Edit `config/patchcore_config.yaml`:

```yaml
model:
  backbone: resnet18           # Options: resnet18, wide_resnet50_2
  layers: [layer2, layer3]     # Feature extraction layers
  coreset_sampling_ratio: 0.1  # Percentage of features to keep
  num_neighbors: 9             # K for K-NN scoring

dataset:
  image_size: [512, 512]       # Input image size
  normalization: imagenet      # Normalization strategy
```

## Troubleshooting

### Training Issues

**"No training data found"**
- Ensure images are in `data/train/good/`
- Check image format (JPG/PNG)
- Verify file permissions

**Out of memory errors**
- Reduce `train_batch_size` in config (try 1)
- Reduce `image_size` (try [256, 256])
- Close other GPU applications

### Validation Issues

**All images predicted as normal (0% defect detection)**
- Need more diverse training data (500+ images recommended)
- Try stronger backbone (wide_resnet50_2)
- Increase coreset_sampling_ratio to 0.2-0.3
- Add layer4 to feature extraction layers
- Ensure defects are clearly visible and significant

**Scores overlap between normal and defective**
- Defects may be too subtle
- Improve image quality and consistency
- Collect more representative normal samples
- Retrain with better quality data

### Deployment Issues

**"No module named anomalib" on Raspberry Pi**
- Good! The standalone script doesn't need Anomalib
- Just install: `pip3 install torch torchvision opencv-python pillow`

**Slow inference on Raspberry Pi**
- Reduce image size in inference
- Use lighter checkpoint (ResNet18)
- Consider quantization for faster CPU inference

**Memory errors on Raspberry Pi**
- Process images one at a time
- Reduce image resolution
- Use 8GB Raspberry Pi 5 model

## Performance Tips

### Improving Model Accuracy
1. **More training data**: 500-1000+ normal images
2. **Stronger backbone**: Switch to wide_resnet50_2
3. **More features**: Add layer4 to layers list
4. **Higher coreset ratio**: Increase to 0.2 or 0.3
5. **Better data quality**: Consistent lighting, positioning

### Optimizing Inference Speed
1. **Smaller backbone**: Use resnet18 (current)
2. **Lower resolution**: Reduce image_size to 256x256
3. **CPU optimization**: Consider quantization
4. **Batch processing**: Process multiple images together

## Key Scripts Reference

| Script | Purpose | Output |
|--------|---------|--------|
| `train.py` | Train PatchCore model | `models/minimal/patchcore_minimal.pth` |
| `validate_model.py` | Basic validation metrics | Console output with accuracy |
| `validate_with_visualization.py` | Detailed validation | CSV + heatmap visualizations |
| `deploy_to_pi.py` | Generate Pi package | `pi_deployment/` directory |

## Environment

The project is tested with:
- **Python**: 3.12.10
- **PyTorch**: 2.9.0 with CUDA 12.1
- **Anomalib**: 2.2.0
- **Lightning**: 2.5.6
- **GPU**: NVIDIA RTX 3080 Ti (12GB)
- **OS**: macOS (also compatible with Linux/Windows)

## References

- [PatchCore Paper](https://arxiv.org/abs/2106.08265) - Original PatchCore research
- [Anomalib Documentation](https://anomalib.readthedocs.io/) - Framework documentation
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) - Training framework
- [Raspberry Pi 5 Documentation](https://www.raspberrypi.com/documentation/)

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Tags

`anomaly-detection` `patchcore` `defect-detection` `pytorch` `raspberry-pi` `computer-vision` `manufacturing` `quality-control` `anomalib`
