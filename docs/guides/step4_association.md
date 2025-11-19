# STEP 4: INTER-CAMERA ASSOCIATION GUIDE

## Mục tiêu
Liên kết tracks từ nhiều cameras để gán **Global ID** cho cùng một người xuất hiện trên nhiều cameras khác nhau.

## Input
- Track features từ Step 3 (`track_features.pkl`) của từng camera
- Có thể từ 2+ cameras

## Output
- `global_id_mapping.json`: Mapping từ local track ID → global ID cho mỗi camera
- `pairwise_matches.json` (Hungarian) hoặc `clusters.json` (Clustering)
- Similarity matrices (nếu visualize)

---

## Phương pháp

### 1. Hungarian Algorithm (Pairwise Matching)
**Ưu điểm:**
- Tối ưu cho từng cặp cameras
- Dễ kiểm soát threshold
- Phù hợp khi có ít cameras (2-4)

**Nhược điểm:**
- Phải xử lý từng cặp
- Có thể có conflict khi merge results

**Sử dụng khi:** 
- Có 2-4 cameras
- Cần kiểm soát chặt chẽ matching quality

### 2. DBSCAN Clustering
**Ưu điểm:**
- Xử lý tất cả cameras cùng lúc
- Tự động tìm số lượng clusters
- Robust với outliers (noise)

**Nhược điểm:**
- Khó tune parameters (eps)
- Có thể merge sai nếu features không tốt

**Sử dụng khi:**
- Có nhiều cameras (4+)
- Features quality tốt
- Muốn automation cao

---

## Cách sử dụng

### Scenario 1: Single camera (testing)
Nếu bạn chỉ có 1 video, có thể test bằng cách chia video thành 2 phần:

```bash
# Split video into 2 parts (giả lập 2 cameras)
ffmpeg -i people-walking.mp4 -ss 00:00:00 -t 00:01:00 -c copy cam1.mp4
ffmpeg -i people-walking.mp4 -ss 00:01:00 -c copy cam2.mp4

# Run Step 2+3 cho mỗi "camera"
python step2_tracking.py --source cam1.mp4 --output-dir runs/cam1_tracking
python step3_reid_extraction.py --source cam1.mp4 \
    --tracks runs/cam1_tracking/tracks/cam1.txt \
    --output-dir runs/cam1_features

python step2_tracking.py --source cam2.mp4 --output-dir runs/cam2_tracking
python step3_reid_extraction.py --source cam2.mp4 \
    --tracks runs/cam2_tracking/tracks/cam2.txt \
    --output-dir runs/cam2_features

# Step 4: Association
python step4_inter_camera_association.py \
    --features runs/cam1_features/track_features.pkl runs/cam2_features/track_features.pkl \
    --camera-names cam1 cam2 \
    --method hungarian \
    --similarity-threshold 0.6 \
    --output-dir runs/step4_mtmc \
    --visualize
```

### Scenario 2: Multiple cameras
```bash
python step4_inter_camera_association.py \
    --features \
        runs/camera1_features/track_features.pkl \
        runs/camera2_features/track_features.pkl \
        runs/camera3_features/track_features.pkl \
    --camera-names camera1 camera2 camera3 \
    --method clustering \
    --eps 0.5 \
    --min-samples 2 \
    --output-dir runs/step4_mtmc \
    --visualize
```

---

## Parameters

### Common
- `--features`: Paths to `track_features.pkl` files (từ Step 3)
- `--camera-names`: Tên cameras (optional, default: cam0, cam1, ...)
- `--method`: `hungarian` hoặc `clustering`
- `--output-dir`: Output directory
- `--visualize`: Tạo similarity matrix heatmaps

### Hungarian Method
- `--similarity-threshold`: Minimum cosine similarity để match (0-1)
  - **0.7-0.8**: Strict (ít false positives)
  - **0.6**: Balanced (recommended)
  - **0.5**: Loose (nhiều matches hơn nhưng có thể sai)

### Clustering Method
- `--eps`: DBSCAN distance threshold
  - Nhỏ hơn = clusters tighter
  - Recommended: 0.4-0.6 cho cosine distance
- `--min-samples`: Minimum tracks per cluster
  - 2 = cho phép 2 tracks từ 2 cameras tạo thành 1 person
  - 3+ = yêu cầu xuất hiện trên 3+ cameras

### Time Constraints (Optional)
- `--time-window`: Maximum frame difference để match
  - Ví dụ: 300 = chỉ match tracks trong cùng 10 giây (30fps)
  - Hữu ích khi cameras có synchronized time

---

## Output Files

### 1. `global_id_mapping.json`
```json
{
  "cam1": {
    "1": 1,    // local track 1 → global ID 1
    "2": 1,    // local track 2 → same person (global ID 1)
    "3": 2     // local track 3 → global ID 2
  },
  "cam2": {
    "1": 1,    // matches with cam1 tracks 1,2
    "2": 3
  }
}
```

### 2. `pairwise_matches.json` (Hungarian)
```json
[
  {
    "cam1": "camera1",
    "track1": 5,
    "cam2": "camera2",
    "track2": 12,
    "similarity": 0.87
  }
]
```

### 3. `clusters.json` (Clustering)
```json
{
  "0": [
    {"camera": "cam1", "track_id": 1},
    {"camera": "cam2", "track_id": 3}
  ],
  "1": [
    {"camera": "cam1", "track_id": 5},
    {"camera": "cam3", "track_id": 2}
  ]
}
```

---

## Tuning Tips

### 1. Check Similarity Matrix
```bash
--visualize
```
Xem heatmap để hiểu distribution:
- Bright diagonal = good features
- Scattered high values = nhiều similar persons
- All low = features không phân biệt được

### 2. Adjust Threshold
**Hungarian method:**
- Bắt đầu với 0.6
- Nếu quá nhiều false matches → tăng lên 0.7-0.8
- Nếu miss too many → giảm xuống 0.5

**Clustering:**
- eps càng nhỏ = clusters càng ít
- Nếu 1 person bị split thành nhiều IDs → tăng eps
- Nếu nhiều người bị merge thành 1 ID → giảm eps

### 3. Feature Quality
Nếu association kém:
- Kiểm tra OSNet weights có load đúng không (Step 3)
- Tăng `--max-crops-per-track` trong Step 3
- Thử train lại OSNet trên domain của bạn

---

## Troubleshooting

### "Need at least 2 cameras for MTMC!"
→ Cần ít nhất 2 track_features.pkl files

### Too many false matches
→ Tăng `--similarity-threshold` (Hungarian) hoặc giảm `--eps` (Clustering)

### Missing matches (người giống nhau không được match)
→ Giảm threshold hoặc tăng eps
→ Check feature quality (visualize embeddings)

### Out of memory
→ Không nên xảy ra vì chỉ làm việc với features (nhỏ)
→ Nếu vẫn bị, giảm số cameras xử lý cùng lúc

---

## Visualization

Similarity matrix heatmap sẽ được lưu ở:
```
runs/step4_mtmc/similarity_cam1_vs_cam2.png
```

**Đọc heatmap:**
- Trục Y: Tracks từ camera 1
- Trục X: Tracks từ camera 2
- Màu sáng (yellow): High similarity → same person
- Màu tối (purple): Low similarity → different persons

---

## Next Steps

Sau khi có global ID mapping:
1. **Visualize results**: Tạo video với global IDs
2. **Evaluate**: So sánh với ground truth (nếu có)
3. **Step 5 (Optional)**: Áp dụng Pose2ID để improve features
4. **Step 6 (Optional)**: Test-time adaptation với PaTTA-ID

---

## Performance Metrics

### Expected Results
- **Good association**: 70-90% của same-person tracks được merge
- **False match rate**: <5%
- **Processing time**: <1 second cho 2 cameras với 200 tracks mỗi camera

### If results are poor:
1. Improve Re-ID features (train OSNet trên domain của bạn)
2. Add temporal constraints (time windows)
3. Add spatial constraints (camera topology)
4. Use appearance + motion fusion
