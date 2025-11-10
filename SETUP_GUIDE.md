# SETUP GUIDE FOR TEAM MEMBERS

## 🎯 Quick Setup (10 minutes)

Follow these steps to set up the project on your machine.

### Prerequisites

- [ ] Python 3.8 or higher installed
- [ ] Git installed
- [ ] NVIDIA GPU with CUDA support (recommended but optional)
- [ ] 8GB+ RAM
- [ ] 20GB+ free disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/mtmc-tracking.git
cd mtmc-tracking
```

### Step 2: Create Virtual Environment

**Option A: Using Conda (Recommended)**
```bash
conda create -n mtmc python=3.10
conda activate mtmc
```

**Option B: Using venv**
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected installation time:** 5-10 minutes

### Step 4: Verify Installation

Run the environment checker:
```bash
python check_environment.py
```

Expected output:
```
✓ Python version: 3.10.x
✓ PyTorch: 2.x.x (CUDA available)
✓ OpenCV: 4.x.x
✓ BoxMOT: 11.x.x
✓ Ultralytics: 8.x.x
✓ torchreid: 2.x.x
```

### Step 5: Download Model Weights

#### YOLOv8 (automatic)
Will download automatically on first run. Or manually:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

#### OSNet Re-ID Model (manual)

**Download link:** (https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)

**File name:** `osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth`

**Instructions:**
1. Download the file from MODEL_ZOO repo (Same-domain ReID: Market1501)
2. Place it in the project root directory:
   ```
   mtmc-tracking/
   ├── osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth
   └── ...
   ```

**Verify download:**
```bash
# Should see the .pth file
ls *.pth
```

### Step 6: Test with Sample Video

Download a test video or use your own:
```bash
# Example: Download from YouTube
youtube-dl -f 'bestvideo[height<=1080]' -o test_video.mp4 YOUR_VIDEO_URL
```

Run a quick test:
```bash
python step2_tracking.py \
    --source test_video.mp4 \
    --output-dir runs/test \
    --device 0
```

### Step 7: Project Structure Check

Your directory should look like:
```
mtmc-tracking/
├── osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth  ← Downloaded
├── yolov8n.pt                                                                                      ← Auto or manual
├── step1_object_detection.py
├── step2_tracking.py
├── step3_reid_extraction.py
├── step4_inter_camera_association.py
├── run_full_mtmc_pipeline.py
├── requirements.txt
├── botsort_config.yaml
└── runs/                                                                                           ← Created automatically
```

---

## 🐛 Troubleshooting

### Issue: "CUDA not available"
**Solution:**
1. Check GPU: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: "ModuleNotFoundError: No module named 'boxmot'"
**Solution:**
```bash
pip install boxmot
```

### Issue: "torchreid not installed" warning
**Solution:**
```bash
pip install torchreid tensorboard
```

### Issue: FFmpeg not found
**Solution:**

**Windows:**
1. Download: https://ffmpeg.org/download.html
2. Extract and add to PATH
3. Verify: `ffmpeg -version`

**Linux:**
```bash
sudo apt install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

### Issue: Out of memory
**Solutions:**
- Reduce batch size
- Use smaller model: `yolov8n.pt` instead of `yolov8x.pt`
- Process shorter videos
- Close other applications

---

## 🚀 Next Steps

1. **Read documentation:**
   - [README.md](README.md) - Overview
   - [STEP1_DETECTION_GUIDE.md](STEP1_DETECTION_GUIDE.md)
   - [STEP2_TRACKING_GUIDE.md](STEP2_TRACKING_GUIDE.md)
   - [STEP3_REID_GUIDE.md](STEP3_REID_GUIDE.md)
   - [STEP4_MTMC_GUIDE.md](STEP4_MTMC_GUIDE.md)

2. **Try examples:**
   ```bash
   # Single camera tracking
   python step2_tracking.py --source video.mp4 --device 0
   
   # Full MTMC pipeline
   python run_full_mtmc_pipeline.py --videos cam1.mp4 cam2.mp4 --device 0
   ```

3. **Configure for your use case:**
   - Edit `botsort_config.yaml` for tracking parameters
   - Adjust thresholds in association scripts
   - Customize output directories

---

## 💻 Development Setup

If you want to contribute or modify the code:

### Install in editable mode
```bash
pip install -e .
```

### Install development dependencies
```bash
pip install pytest black flake8 mypy
```

### Run tests
```bash
pytest tests/
```

### Code formatting
```bash
black .
flake8 .
```

---

## 📞 Getting Help

- **Issues:** Open a GitHub issue
- **Questions:** Check existing issues or documentation
- **Team chat:** [Your team communication channel]

---

## ✅ Setup Checklist

- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`requirements.txt`)
- [ ] Environment verified (`check_environment.py`)
- [ ] YOLOv8 model downloaded
- [ ] OSNet model downloaded and placed correctly
- [ ] Test run completed successfully
- [ ] Documentation read
- [ ] Ready to start tracking! 🎉

---

**Estimated total setup time:** 15-20 minutes (including downloads)

**Need help?** Contact the team or open a GitHub issue.
