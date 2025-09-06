# PatchCore Defect Detection System

A complete solution for parts defect detection using PatchCore anomaly detection with high-resolution training and tiling strategy, designed for deployment on Raspberry Pi 5.

## 🎯 Features

- **PatchCore Model**: State-of-the-art anomaly detection using Anomalib
- **High-Resolution Training**: Supports 1024x1024+ images
- **Tiling Strategy**: Efficient inference on large images with overlap stitching
- **Raspberry Pi Deployment**: Optimized for Raspberry Pi 5 with camera
- **CSV Database**: Automatic logging of all detections
- **Real-time Monitoring**: Continuous defect detection with configurable intervals
- **Visualization**: Anomaly heatmaps and result overlays

## 🏗️ Project Structure

```
patchcore_defect_detection/
├── data/                          # Training and test data
│   ├── train/good/               # Normal training images
│   ├── test/good/                # Normal test images
│   ├── test/defective/           # Defective test images
│   └── val/                      # Validation data
├── models/                       # Trained models
│   ├── saved_models/             # Training outputs
│   └── deployment/               # Deployment-ready models
├── src/                          # Source code
│   ├── training/                 # Training scripts
│   ├── inference/                # Inference and tiling
│   ├── utils/                    # Utilities
│   └── deployment/               # Raspberry Pi deployment
├── config/                       # Configuration files
├── database/                     # Detection results and images
└── logs/                         # Training and inference logs
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or create project directory
mkdir patchcore_defect_detection
cd patchcore_defect_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_local.txt
```

### 2. Data Preparation

Place your images in the following structure:
```
data/
├── train/
│   └── good/              # Only normal/good parts (50-100+ images)
├── test/
│   ├── good/              # Normal test images (20+ images)
│   └── defective/         # Defective test images (20+ images)
```

**Image Requirements:**
- High resolution (1024x1024 or higher recommended)
- Supported formats: JPG, PNG
- Consistent lighting and positioning
- Clear, focused images

### 3. Training

```bash
# Run training
python train.py
```

The training will:
- Load configuration from `config/patchcore_config.yaml`
- Train PatchCore model on normal images only
- Evaluate on test set
- Save model to `models/deployment/`
- Generate training logs and metrics

### 4. Local Testing

```bash
# Test on single image
python -c "
from src.inference.tiled_inference import TiledInference
import cv2

# Load model
detector = TiledInference('models/deployment/patchcore_deployment.pth')

# Load and test image
image = cv2.imread('path/to/test/image.jpg')
results = detector.predict(image)

print(f'Anomaly Score: {results[\"anomaly_score\"]:.4f}')
print(f'Is Defective: {results[\"is_anomaly\"]}')
"
```

### 5. Raspberry Pi Deployment

```bash
# Prepare deployment package
python deploy_to_pi.py --model_path models/deployment/patchcore_deployment.pth --pi_address 192.168.1.100

# Or create manual deployment package
python deploy_to_pi.py --model_path models/deployment/patchcore_deployment.pth
```

### 🔧 Configuration

### Data Organization Guidelines

    - **Training Data**: Place only normal/good parts in data/train/good/
    - **Test Data**: Place both normal and defective parts in respective folders
    - **Image Format**: Use high-resolution images (recommended 1024x1024 or higher)
    - **Naming Convention**: Use descriptive names like part_001_normal.jpg, part_002_defective.jpg
    - **Minimum Images**: At least 50-100 normal images for training, 20-30 for testing each class


### Model Configuration (`config/patchcore_config.yaml`)

```yaml
model:
  backbone: wide_resnet50_2        # Feature extractor
  layers: [layer2, layer3]         # Feature layers
  coreset_sampling_ratio: 0.1      # Memory bank sampling
  num_neighbors: 9                 # k-NN neighbors

dataset:
  image_size: [1024, 1024]         # High-resolution training
  
tiling:
  tile_size: [256, 256]            # Inference tile size
  stride: [128, 128]               # 50% overlap
```

### Detection Parameters

- **Confidence Threshold**: Adjust in deployment script (default: 0.5)
- **Capture Interval**: Time between detections (default: 5 seconds)
- **Tile Size**: Balance between accuracy and speed
- **Stride**: Overlap for better coverage

## 🎮 Raspberry Pi Usage

### Manual Control

```python
from src.deployment.raspberry_pi_detector import RaspberryPiDetector

# Initialize detector
detector = RaspberryPiDetector(
    model_path="models/patchcore_model.pth",
    csv_path="database/detection_results.csv",
    confidence_threshold=0.5
)

# Single image detection
image = detector.camera.capture_image()
results = detector.detect_single_image(image)

# Continuous monitoring
detector.continuous_monitoring(
    capture_interval=5.0,
    save_images=True,
    max_detections=100
)

# Batch processing
detector.batch_process_directory("input/images/", "output/results/")
```

### Service Mode

```bash
# Start as systemd service (after deployment)
sudo systemctl start patchcore_detector
sudo systemctl status patchcore_detector

# View logs
journalctl -u patchcore_detector -f

# Stop service
sudo systemctl stop patchcore_detector
```

## 📊 Database Schema

Detection results are automatically saved to `database/detection_results.csv`:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Detection timestamp |
| anomaly_score | float | Anomaly confidence (0-1) |
| is_defective | boolean | Defect detected flag |
| confidence_threshold | float | Threshold used |
| image_shape | string | Image dimensions |
| num_tiles_processed | int | Number of tiles |
| detection_status | string | NORMAL/DEFECTIVE |

## 📈 Monitoring and Analytics

### Get Statistics

```python
# Get recent statistics
stats = detector.get_detection_statistics(days=7)
print(f"Defect rate: {stats['defect_rate']:.2f}%")
print(f"Total detections: {stats['total_detections']}")
```

### Export Reports

```python
# Export detection report
detector.csv_handler.export_report("reports/monthly_report.csv", days=30)
```

### Visualizations

The system automatically creates:
- Anomaly heatmaps overlaid on original images
- Detection status indicators
- Statistical summaries and trends

## 🔄 Model Retraining

To retrain with new data:

1. Add new normal images to `data/train/good/`
2. Add new test cases to `data/test/`
3. Run training: `python train.py`
4. Deploy updated model: `python deploy_to_pi.py --model_path models/deployment/patchcore_deployment.pth`

## 🛠️ Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# Check camera status
vcgencmd get_camera

# Enable camera
sudo raspi-config
```

**Model loading errors:**
- Ensure PyTorch version compatibility
- Check model file paths
- Verify CUDA/CPU device settings

**Poor detection accuracy:**
- Increase training data (especially normal samples)
- Adjust confidence threshold
- Improve image quality and consistency
- Fine-tune tiling parameters

**Performance issues:**
- Reduce image resolution
- Increase tile stride (less overlap)
- Enable quantization in deployment config

### Logs and Debugging

```bash
# Check system logs
tail -f logs/detector.log

# Monitor system resources
htop

# Python debugging
export PYTHONPATH=/path/to/project
python -m pdb script.py
```

## 🔧 Hardware Requirements

### Local Training Machine
- GPU recommended (NVIDIA with CUDA support)
- 16GB+ RAM
- 50GB+ storage
- Python 3.8+

### Raspberry Pi 5
- Raspberry Pi 5 (4GB/8GB recommended)
- MicroSD card (64GB+ Class 10)
- Raspberry Pi Camera Module or USB camera
- Adequate power supply (official recommended)

## 📚 Dependencies

### Local Training
- PyTorch ≥ 2.0.0
- Anomalib ≥ 1.0.0
- OpenCV ≥ 4.8.0
- Lightning ≥ 2.0.0

### Raspberry Pi
- PyTorch (CPU) ≥ 2.0.0
- OpenCV ≥ 4.8.0
- Picamera2 ≥ 0.3.0
- NumPy, Pandas

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Ensure hardware compatibility
4. Verify data format and quality

## 🔗 References

- [Anomalib Documentation](https://anomalib.readthedocs.io/)
- [PatchCore Paper](https://arxiv.org/abs/2106.08265)
- [Raspberry Pi Camera Guide](https://www.raspberrypi.org/documentation/accessories/camera.html)