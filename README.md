# Multi-Target Multi-Camera Tracking (MTMC) Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete end-to-end pipeline for Multi-Target Multi-Camera Tracking using **YOLOv8**, **BoT-SORT**, and **OSNet Re-ID**.

## 🎯 Overview

This pipeline implements a full MTMC (Multi-Target Multi-Camera) tracking system with the following steps:

1. **Object Detection** - YOLOv8 person detection
2. **Single-Camera Tracking** - BoT-SORT tracker with ReID features
3. **Re-ID Feature Extraction** - OSNet pretrained model
4. **Inter-Camera Association** - Hungarian algorithm or DBSCAN clustering

## ✨ Features

- ✅ **Complete Pipeline**: From detection to global ID assignment
- ✅ **Memory Efficient**: Handles long videos without OOM errors
- ✅ **Flexible**: Support multiple association methods
- ✅ **Production Ready**: Batch processing scripts included
- ✅ **Well Documented**: Step-by-step guides for each component

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- FFmpeg (optional, for video splitting)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mtmc-tracking.git
cd mtmc-tracking
```

### 2. Create virtual environment

```bash
# Using conda (recommended)
conda create -n mtmc python=3.10
conda activate mtmc

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download pretrained models

#### YOLOv8 (auto-downloaded on first run)
```bash
# Will download automatically when running step2_tracking.py
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

#### OSNet Re-ID Model
```bash
# Download from MODEL_ZOO repo
# Place in project root directory
```

**Option 1: Market-1501 pretrained (recommended)**
- Download: [osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)
- Size: ~2.2M parameters
- Performance: 94.8% mAP on Market-1501

**Option 2: Use your own weights**
- Train with FastReID: https://github.com/JDAI-CV/fast-reid
- Or use any compatible OSNet weights

Place the `.pth` file in the project root:
```
mtmc-tracking/
├── osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth
├── yolov8n.pt
└── ...
```

## 📖 Usage

### Quick Start (Full Pipeline)

```bash
python run_full_mtmc_pipeline.py \
    --videos video1.mp4 video2.mp4 video3.mp4 \
    --camera-names cam1 cam2 cam3 \
    --device 0 \
    --method hungarian \
    --similarity-threshold 0.6
```

### Step-by-Step Usage

#### Step 1: Object Detection (Optional)
```bash
python step1_object_detection.py \
    --source video.mp4 \
    --output-dir runs/detection \
    --save-txt \
    --device 0
```

#### Step 2: Single-Camera Tracking
```bash
python step2_tracking.py \
    --source video.mp4 \
    --output-dir runs/tracking \
    --device 0
```

#### Step 3: Re-ID Feature Extraction
```bash
python step3_reid_extraction.py \
    --source video.mp4 \
    --tracks runs/tracking/exp/tracks/video.txt \
    --output-dir runs/features \
    --device 0 \
    --max-crops-per-track 10
```

#### Step 4: Inter-Camera Association
```bash
python step4_inter_camera_association.py \
    --features runs/cam1_features/track_features.pkl runs/cam2_features/track_features.pkl \
    --camera-names cam1 cam2 \
    --method hungarian \
    --similarity-threshold 0.6 \
    --output-dir runs/mtmc_results \
    --visualize
```

## 📁 Project Structure

```
mtmc-tracking/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── step1_object_detection.py          # YOLO detection
├── step2_tracking.py                  # BoT-SORT tracking
├── step2_tracking_from_step1.py       # Tracking from detections
├── step3_reid_extraction.py           # Re-ID feature extraction
├── step4_inter_camera_association.py  # MTMC association
│
├── run_full_mtmc_pipeline.py          # Automated pipeline
├── run_mtmc_pipeline.bat              # Windows batch script
├── run_mtmc_pipeline.sh               # Linux bash script
│
├── botsort_config.yaml                # BoT-SORT configuration
│
├── check_environment.py               # Environment validator
├── check_boxmot_env.py               # BoxMOT setup checker
├── fix_torchreid.py                  # Debug torchreid issues
├── simulate_multicam.py              # Split video for testing
│
├── STEP1_DETECTION_GUIDE.md          # Step 1 documentation
├── STEP2_TRACKING_GUIDE.md           # Step 2 documentation
├── STEP3_REID_GUIDE.md               # Step 3 documentation
├── STEP4_MTMC_GUIDE.md               # Step 4 documentation
├── README_MTMC_BoxMOT.md             # Original overview
│
└── runs/                              # Output directory (git-ignored)
    ├── detection/
    ├── tracking/
    ├── features/
    └── mtmc_results/
```

## 🔧 Configuration

### BoT-SORT Tracker
Edit `botsort_config.yaml` to adjust tracking parameters:
- `track_high_thresh`: 0.5 (detection confidence)
- `match_thresh`: 0.8 (IoU threshold)
- `with_reid`: true (enable ReID)

### Re-ID Feature Extraction
- `--max-crops-per-track`: Number of frames to sample per track
- `--reid-size`: Input size for OSNet [256, 128]

### Association Methods

**Hungarian Algorithm** (recommended for 2-4 cameras):
```bash
--method hungarian --similarity-threshold 0.6
```

**DBSCAN Clustering** (good for 4+ cameras):
```bash
--method clustering --eps 0.5 --min-samples 2
```

## 📊 Output Format

### Global ID Mapping (`global_id_mapping.json`)
```json
{
  "cam1": {
    "1": 1,    // local track 1 → global ID 1
    "2": 2,
    "3": 1     // local track 3 → same person (global ID 1)
  },
  "cam2": {
    "1": 1,    // matches with cam1
    "2": 3
  }
}
```

### Track Features (`track_features.pkl`)
```python
{
    track_id: {
        'features': np.array([512,]),      # Average feature vector
        'all_features': np.array([N, 512]), # All frame features
        'frame_indices': [1, 5, 10, ...]   # Frame numbers
    }
}
```

## 🎨 Visualization

Similarity matrices are automatically generated with `--visualize`:

```bash
runs/mtmc_results/similarity_cam1_vs_cam2.png
```

Heatmap shows cosine similarity between all track pairs.

## 🐛 Troubleshooting

### "Out of Memory" error in Step 3
- Reduce `--max-crops-per-track` (default: 10)
- Process shorter video segments
- Use CPU instead of GPU for feature extraction

### Poor association results
1. Check OSNet weights loaded correctly
2. Visualize similarity matrices (`--visualize`)
3. Adjust threshold/eps parameters
4. Increase `--max-crops-per-track` for better features

### "torchreid not installed" warning
```bash
pip install torchreid
# If still issues, install tensorboard:
pip install tensorboard
```

### BoxMOT import errors
```bash
pip install --upgrade boxmot
```

## 📈 Performance

**Hardware:** NVIDIA RTX 3080 (10GB)

| Step | Input | Speed | Output |
|------|-------|-------|--------|
| Detection | 1920x1080 @ 30fps | ~35 FPS | Bounding boxes |
| Tracking | 9000 frames | ~40 FPS | 263 tracks |
| Feature Extraction | 263 tracks | ~2 min | 512-dim features |
| Association | 2 cameras | <1 sec | Global IDs |

## 🔬 Advanced Features

### Time-based constraints
Only match tracks within time window:
```bash
--time-window 300  # 300 frames ≈ 10 seconds
```

### Batch processing
Use provided scripts for multiple cameras:
```bash
./run_mtmc_pipeline.sh  # Linux
run_mtmc_pipeline.bat   # Windows
```

### Custom ReID models
Replace OSNet with your trained model:
```bash
python step3_reid_extraction.py \
    --reid-model /path/to/your/model.pth
```

## 📚 Documentation

Detailed guides for each step:
- [STEP1_DETECTION_GUIDE.md](STEP1_DETECTION_GUIDE.md) - Object detection
- [STEP2_TRACKING_GUIDE.md](STEP2_TRACKING_GUIDE.md) - Single-camera tracking
- [STEP3_REID_GUIDE.md](STEP3_REID_GUIDE.md) - Feature extraction
- [STEP4_MTMC_GUIDE.md](STEP4_MTMC_GUIDE.md) - Multi-camera association

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BoxMOT](https://github.com/mikel-brostrom/boxmot) - Multi-object tracking library
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [FastReID](https://github.com/JDAI-CV/fast-reid) - Re-identification training
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Tracker with ReID

## 📞 Contact

For questions or issues, please open a GitHub issue or contact the team.

## 🗺️ Roadmap

- [ ] Pose2ID integration (CVPR 2025)
- [ ] PaTTA-ID domain adaptation
- [ ] Real-time streaming support
- [ ] Web-based visualization dashboard
- [ ] Docker container
- [ ] Pre-trained models on custom datasets

## 📊 Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{mtmc-tracking-2025,
  author = {Your Team},
  title = {Multi-Target Multi-Camera Tracking Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/mtmc-tracking}
}
```

---

**⭐ Star this repo if you find it helpful!**
