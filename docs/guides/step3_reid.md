# BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG Re-ID VÀ HUẤN LUYỆN OSNet

## Tổng quan

Bước 3 gồm 2 phần chính:
1. **Trích xuất đặc trưng Re-ID** từ tracks (bắt buộc cho MTMC)
2. **Huấn luyện OSNet với FastReID** (optional - có thể dùng pretrained)

## Phần A: Trích xuất đặc trưng từ Tracks

### Mục đích
- Crop người từ video dựa trên tracks (Step 2)
- Extract features vector bằng OSNet pretrained
- Lưu features cho Inter-Camera Association (Step 4)

### Chạy extraction

```bash
python step3_reid_extraction.py \
  --source people-walking.mp4 \
  --tracks runs/step2_tracking/exp/tracks/people-walking.txt \
  --reid-model osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth \
  --output-dir runs/step3_reid_features \
  --save-crops \
  --max-crops-per-track 10
```

### Tham số quan trọng

- `--save-crops`: Lưu crops để visualize hoặc train model mới
- `--max-crops-per-track`: Giới hạn số crops (sample đều), default=10
- `--reid-size`: Kích thước input cho OSNet [256, 128]

### Output

```
runs/step3_reid_features/
├── track_features.pkl          # Features dict: {track_id: {...}}
├── track_metadata.json         # Metadata: frame indices, num frames
└── crops/                      # (nếu --save-crops)
    ├── track_0001/
    │   ├── frame_000005_000.jpg
    │   ├── frame_000010_001.jpg
    │   └── ...
    └── track_0002/
        └── ...
```

### Feature format

```python
# track_features.pkl structure
{
    track_id: {
        'features': np.array([...]),        # Averaged feature vector
        'all_features': np.array([[...]]),  # All individual features
        'frame_indices': [5, 10, 15, ...]   # Frame numbers
    }
}
```

## Phần B: Huấn luyện OSNet với FastReID (Optional)

Nếu muốn train model OSNet mới trên dữ liệu của bạn:

### 1. Cài đặt FastReID

```bash
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
pip install -r requirements.txt
python setup.py develop
```

### 2. Chuẩn bị dữ liệu

Cần cấu trúc thư mục:

```
dataset/
├── train/
│   ├── 0001/  # ID 1
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── 0002/  # ID 2
│   │   └── ...
│   └── ...
├── query/
│   └── ...
└── gallery/
    └── ...
```

**Có thể dùng crops từ Step 3A** (nếu có labels đúng):

```bash
# Copy crops sang format FastReID
python prepare_fastreid_data.py \
  --crops-dir runs/step3_reid_features/crops \
  --output-dir dataset/my_dataset
```

### 3. Config file

Tạo `configs/my_osnet.yml`:

```yaml
MODEL:
  BACKBONE:
    NAME: "osnet_x1_0"
  HEADS:
    NAME: "BNneckHead"
    POOL_LAYER: "avgpool"
  
DATASETS:
  NAMES: ("my_dataset",)
  TESTS: ("my_dataset",)

DATALOADER:
  NUM_WORKERS: 4

SOLVER:
  OPT: "Adam"
  BASE_LR: 0.00035
  MAX_EPOCH: 120
  WARMUP_ITERS: 1500

INPUT:
  SIZE_TRAIN: [256, 128]
  SIZE_TEST: [256, 128]

OUTPUT_DIR: "logs/osnet_my_dataset"
```

### 4. Training

```bash
cd fast-reid
python tools/train_net.py \
  --config-file configs/my_osnet.yml \
  --num-gpus 1
```

### 5. Export weights

```bash
# Model sẽ ở logs/osnet_my_dataset/model_final.pth
# Copy về workspace
cp logs/osnet_my_dataset/model_final.pth /path/to/boxmot-test/my_osnet.pth
```

## Phần C: Tích hợp Pose2ID (Training-free enhancement)

Pose2ID tăng độ chính xác ReID bằng cách:
- Sinh ảnh với nhiều poses khác nhau
- "Feature Centralization" để vector đặc trưng ổn định

### 1. Clone Pose2ID

```bash
git clone https://github.com/yuanc3/Pose2ID.git
cd Pose2ID
pip install -r requirements.txt
```

### 2. Apply Pose2ID lên features

```python
# pose2id_enhance.py (pseudo-code)
from pose2id import FeatureCentralizer

# Load features từ Step 3A
features = pickle.load(open('track_features.pkl', 'rb'))

# Apply Pose2ID
centralizer = FeatureCentralizer()
enhanced_features = {}

for track_id, data in features.items():
    enhanced = centralizer.centralize(data['all_features'])
    enhanced_features[track_id] = enhanced

# Save enhanced features
pickle.dump(enhanced_features, open('enhanced_features.pkl', 'wb'))
```

## Tối ưu hóa

### 1. Tăng tốc inference

Export OSNet sang ONNX:

```python
import torch
from torchreid.models import build_model

model = build_model('osnet_x1_0', num_classes=1000)
model.load_state_dict(torch.load('osnet.pth'))
model.eval()

dummy_input = torch.randn(1, 3, 256, 128)
torch.onnx.export(model, dummy_input, "osnet.onnx")
```

### 2. Giảm feature dimension

Dùng PCA để giảm từ 512 → 128 dims:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=128)
reduced_features = pca.fit_transform(all_features)
```

### 3. Batch processing

Xử lý nhiều crops cùng lúc để tận dụng GPU:

```python
batch_size = 32
for i in range(0, len(crops), batch_size):
    batch = crops[i:i+batch_size]
    features = model(batch)
```

## Dependencies

Cài thêm các thư viện cho Step 3:

```bash
# Core
pip install torchreid

# Optional: FastReID
pip install yacs
pip install faiss-gpu  # hoặc faiss-cpu

# Optional: Pose2ID
# Follow Pose2ID repo instructions
```

## Troubleshooting

### 1. torchreid không cài được

```bash
# Clone và cài từ source
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -e .
```

### 2. Out of memory khi extract features

- Giảm `--max-crops-per-track`
- Giảm batch size trong code
- Dùng CPU: `--device cpu`

### 3. Features không tốt

- Kiểm tra pretrained model có đúng dataset không (Market-1501, DukeMTMC, ...)
- Thử normalize features: `features = features / np.linalg.norm(features)`
- Áp dụng Pose2ID

## Workflow đầy đủ

```bash
# 1. Extract features với pretrained OSNet
python step3_reid_extraction.py \
  --source video.mp4 \
  --tracks runs/step2_tracking/exp/tracks/video.txt \
  --save-crops \
  --output-dir runs/step3_reid

# 2. (Optional) Train custom OSNet
cd fast-reid
python tools/train_net.py --config-file configs/my_osnet.yml

# 3. (Optional) Re-extract với custom model
python step3_reid_extraction.py \
  --reid-model my_osnet.pth \
  ...

# 4. (Optional) Apply Pose2ID enhancement
python apply_pose2id.py \
  --features runs/step3_reid/track_features.pkl \
  --output runs/step3_reid/enhanced_features.pkl
```

## Kết quả mong đợi

```
EXTRACTION SUMMARY
======================================================================
Total tracks:       15
Feature dimension:  512
Output directory:   runs/step3_reid_features
Crops saved:        runs/step3_reid_features/crops
======================================================================
```

Features này sẽ được dùng cho **Bước 4: Inter-Camera Association** để liên kết người giữa các cameras.

## Next Steps

Sau khi có features:
1. ✅ Visualize features bằng t-SNE/UMAP
2. ✅ Tính similarity matrix giữa tracks
3. ➡️ **Chuyển sang Bước 4**: Inter-Camera Association

## Tham khảo

- [FastReID Docs](https://github.com/JDAI-CV/fast-reid)
- [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- [Pose2ID Paper](https://github.com/yuanc3/Pose2ID)
- [OSNet Paper](https://arxiv.org/abs/1905.00953)
