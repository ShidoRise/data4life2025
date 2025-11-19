# Quick Start Guide

Get up and running in 5 minutes!

## 🚀 Installation

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/mtmc-tracking.git
cd mtmc-tracking

# Create environment
conda create -n mtmc python=3.10
conda activate mtmc

# Install dependencies
pip install -r requirements.txt
```

## 📥 Download Models

```bash
cd models/

# YOLOv8 (will auto-download on first run)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# OSNet Re-ID - Download from Google Drive:
# https://drive.google.com/file/d/1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY/view
# Place the downloaded .pth file in models/
```

## 🎬 Prepare Data

```bash
# Place your videos in data/ folder
cp /path/to/camera1.mp4 data/
cp /path/to/camera2.mp4 data/
```

## ▶️ Run Pipeline

```bash
# Full MTMC pipeline (all cameras)
python pipeline/run_mtmc.py \
    --videos data/camera1.mp4 data/camera2.mp4 \
    --camera-names cam1 cam2 \
    --device 0 \
    --output-dir outputs/experiment1
```

## 📊 View Results

```bash
# Results in:
outputs/experiment1/mtmc_results/
├── global_id_mapping.json    # Final global IDs
├── pairwise_matches.json     # Association details
└── similarity_cam1_cam2.png  # Visualization
```

## 🔍 Step-by-Step (Optional)

If you want to run each step individually:

```bash
# Step 2: Single-camera tracking
python scripts/step2_tracking.py \
    --source data/camera1.mp4 \
    --project outputs \
    --name cam1_tracking \
    --device 0 \
    --save-txt

# Step 3: Extract Re-ID features
python scripts/step3_reid_extraction.py \
    --source data/camera1.mp4 \
    --tracks outputs/cam1_tracking/tracks/camera1.txt \
    --output-dir outputs/cam1_features \
    --device 0

# Step 4: Multi-camera association
python scripts/step4_association.py \
    --features outputs/cam1_features/track_features.pkl \
               outputs/cam2_features/track_features.pkl \
    --camera-names cam1 cam2 \
    --method hungarian \
    --output-dir outputs/mtmc_results \
    --visualize
```

## ✅ Verify Setup

```bash
python tools/check_environment.py
```

Should show:
```
✓ Python version: 3.10.x
✓ PyTorch: 2.x.x (CUDA available)
✓ BoxMOT: 11.x.x
✓ All dependencies installed
```

## 📚 Next Steps

- Read full documentation: `docs/SETUP_GUIDE.md`
- Step-by-step guides: `docs/guides/`
- Customize configs: `configs/botsort_config.yaml`

## 🆘 Troubleshooting

**CUDA not available?**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Missing OSNet model?**
- Download from Google Drive link above
- Place in `models/` directory
- Verify: `ls models/*.pth`

**Need help?**
- Check `docs/SETUP_GUIDE.md` for detailed setup
- See `REORGANIZATION.md` for project structure
- Open GitHub issue if stuck

---

**That's it! You're ready to track!** 🎉
