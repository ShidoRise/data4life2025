# Models Directory

Place your pretrained model weights here:

## Required Models

### 1. YOLOv8 Detection Model
- **File**: `yolov8n.pt` (or yolov8s.pt, yolov8m.pt, etc.)
- **Download**: Auto-downloads on first run, or manually from:
  - https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

### 2. OSNet Re-ID Model
- **File**: `osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth`
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY/view)
- **Alternative short name**: You can rename to `osnet_market.pth` for convenience

## Model Directory Structure

```
models/
├── yolov8n.pt                    # Detection model
├── osnet_market.pth              # Re-ID model (renamed for convenience)
└── README.md                     # This file
```

## Notes

- Model files are gitignored (too large for git)
- Each team member needs to download models independently
- See main README.md for detailed setup instructions
