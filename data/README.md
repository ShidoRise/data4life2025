# Data Directory

Place your input videos here for MTMC tracking.

## Directory Structure

```
data/
├── camera1.mp4       # Video from camera 1
├── camera2.mp4       # Video from camera 2
├── camera3.mp4       # Video from camera 3
└── README.md         # This file
```

## Supported Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)

## Usage

```bash
# Single camera
python scripts/step2_tracking.py --source data/camera1.mp4

# Multiple cameras
python pipeline/run_mtmc.py --videos data/camera1.mp4 data/camera2.mp4
```

## Notes

- Video files are gitignored (too large)
- Use your own multi-camera dataset
- Videos should have overlapping views for better association
