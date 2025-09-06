# Data Organization Guide

## Training Data Structure

Place your images in the following structure:

```
data/
├── train/
│   └── good/              # ONLY normal/good parts (50-100+ images)
│       ├── part001.jpg
│       ├── part002.jpg
│       └── ...
├── test/
│   ├── good/              # Normal test images (20+ images)
│   │   ├── test_normal_001.jpg
│   │   └── ...
│   └── defective/         # Defective test images (20+ images)
│       ├── test_defect_001.jpg
│       └── ...
└── val/                   # Optional validation set
    ├── good/
    └── defective/
```

## Image Requirements for RTX 3080 Ti Training

- **Resolution**: 1024x1024 pixels or higher (your GPU can handle it!)
- **Format**: JPG, PNG supported
- **Quality**: High quality, well-lit, focused images
- **Consistency**: Similar lighting, angle, background
- **Quantity**: 
  - Training (normal): 50-100+ images minimum
  - Test (normal): 20+ images
  - Test (defective): 20+ images

## Tips for Best Results

1. **More normal data is better** - PatchCore learns what "normal" looks like
2. **Consistent imaging conditions** - same lighting, distance, angle
3. **High resolution** - your RTX 3080 Ti can handle 1024x1024 easily
4. **Variety in normal samples** - different orientations, minor variations
5. **Clear defects in test data** - obvious defects for evaluation

## Next Steps

1. Add your images to the appropriate folders
2. Run: `python train.py`
3. Monitor training with: `tensorboard --logdir logs/tensorboard`
