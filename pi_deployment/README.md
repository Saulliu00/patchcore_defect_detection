# PatchCore Inference for Raspberry Pi

## Package Contents

- `pi_inference.py` - Standalone inference script (no Anomalib required!)
- `patchcore_pi.pth` - Optimized model checkpoint
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Installation on Raspberry Pi

### 1. Transfer Files

Copy this entire folder to your Raspberry Pi:

```bash
scp -r pi_deployment/ pi@<PI_IP>:/home/pi/patchcore/
```

### 2. Install Dependencies

SSH into your Pi and install dependencies:

```bash
cd /home/pi/patchcore
python3 -m pip install -r requirements.txt
```

**Note:** PyTorch installation on Pi might take 10-30 minutes.

### 3. Prepare Your Images

Place images in a folder, for example:

```bash
mkdir -p /home/pi/data/val
# Copy your test images here
```

## Usage

### Basic Usage

Process all images in a folder:

```bash
python3 pi_inference.py --input /home/pi/data/val --output results.csv
```

### Advanced Usage

```bash
python3 pi_inference.py \
    --input /path/to/images \
    --output detailed_result.csv \
    --checkpoint patchcore_pi.pth \
    --threshold 0.95 \
    --device cpu
```

### Arguments

- `--input`: Folder containing images to process (required)
- `--output`: Output CSV file path (default: detailed_result.csv)
- `--checkpoint`: Model checkpoint file (default: patchcore_pi.pth)
- `--threshold`: Anomaly score threshold (default: 0.95)
- `--device`: cpu or cuda (default: cpu)

## Output Format

The script generates a CSV file with these columns:

| Column | Description |
|--------|-------------|
| filename | Image filename |
| score | Anomaly score (higher = more anomalous) |
| predicted_label | 'normal' or 'defective' |
| path | Relative path from input folder |

### Example Output

```csv
filename,score,predicted_label,path
defect_01.jpg,0.9823,defective,defect_01.jpg
normal_01.jpg,0.8234,normal,normal_01.jpg
defect_02.jpg,0.9654,defective,defect_02.jpg
```

## Threshold Tuning

The default threshold is **0.95**. Adjust based on your needs:

- **Higher threshold (0.98)**: Fewer false positives, may miss some defects
- **Lower threshold (0.90)**: Catches more defects, more false positives
- **Optimal**: Experiment with your validation data

## Performance

On Raspberry Pi 5:
- **Processing speed**: ~2-5 seconds per image (512x512)
- **Memory usage**: ~500MB RAM
- **Model size**: ~45MB

## Troubleshooting

### "No module named 'torch'"
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "Memory bank not found"
The model needs to be properly trained with memory bank. Retrain using train.py.

### Slow inference
- Reduce image size in training config
- Use smaller batch of images
- Consider using Raspberry Pi 5 for better performance

## Integration with Camera

To capture images from Pi Camera and run inference:

```python
from picamera2 import Picamera2
from pi_inference import PatchCoreInference

camera = Picamera2()
camera.start()

inference = PatchCoreInference('patchcore_pi.pth')

# Capture and process
camera.capture_file('/tmp/test.jpg')
result = inference.predict('/tmp/test.jpg')
print(f"Score: {result['score']}, Prediction: {result['prediction']}")
```

## Support

For issues or questions:
1. Check model was trained successfully
2. Verify all dependencies installed
3. Test with known good/bad images first
