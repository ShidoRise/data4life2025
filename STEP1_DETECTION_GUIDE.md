# BƯỚC 1: OBJECT DETECTION VỚI YOLOv8

## Tổng quan

Bước đầu tiên trong pipeline MTMC là phát hiện đối tượng (người) trong video từ mỗi camera. Chúng ta sử dụng **YOLOv8n** (nano version) - phiên bản nhẹ nhất của YOLOv8, phù hợp cho triển khai trên thiết bị yếu.

## Tính năng chính

### 1. **PersonDetector Class**
- Wrapper cho YOLOv8 model, chuyên biệt hóa cho việc phát hiện người
- Hỗ trợ điều chỉnh confidence threshold và IoU threshold
- Chỉ detect class "person" (class 0 trong COCO dataset)

### 2. **Xử lý Video**
```python
detector = PersonDetector(model_path='yolov8n.pt', conf_threshold=0.25)
stats = detector.detect_video(
    video_source='video.mp4',
    output_dir='runs/detection',
    save_txt=True,      # Lưu detection cho bước tracking
    save_video=True,    # Lưu video đã annotate
    show=False          # Hiển thị real-time
)
```

### 3. **Output Format**

#### Text Files (YOLO Format)
Mỗi frame tạo ra một file `.txt` với format:
```
class x_center y_center width height confidence
0 0.512345 0.345678 0.123456 0.234567 0.856789
0 0.678901 0.456789 0.098765 0.187654 0.765432
```

- `class`: luôn là 0 (person)
- `x_center, y_center`: tọa độ trung tâm bbox (normalized 0-1)
- `width, height`: kích thước bbox (normalized 0-1)
- `confidence`: độ tin cậy của detection (0-1)

#### Video Output
- Video đã annotate với bounding boxes màu xanh
- Hiển thị thông tin: frame number, số người, FPS

### 4. **Thống kê**
```
KẾT QUẢ DETECTION
======================================================================
Tổng số frames xử lý:          1247
Tổng số người phát hiện:       3891
Trung bình người/frame:        3.12
Thời gian xử lý TB/frame:      45.23 ms
FPS trung bình:                22.11
Kết quả lưu tại:               runs/step1_detection
======================================================================
```

## Hướng dẫn sử dụng

### Yêu cầu
```bash
# Kích hoạt environment
conda activate boxmot

# Kiểm tra các thư viện đã cài
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
```

### Chạy detection cơ bản
```bash
python step1_object_detection.py
```

### Tùy chỉnh parameters trong code

```python
# Điều chỉnh ngưỡng confidence
detector = PersonDetector(
    model_path='yolov8n.pt',
    conf_threshold=0.3,   # Tăng lên để giảm false positives
    iou_threshold=0.5     # Điều chỉnh NMS
)

# Detect từ webcam
detector.detect_video(video_source='0', show=True)

# Detect từ thư mục ảnh
detector.detect_images(
    image_dir='path/to/images',
    output_dir='runs/image_detection'
)
```

## Cấu trúc Output

```
runs/step1_detection/
├── labels/                    # Detection text files
│   ├── frame_000001.txt
│   ├── frame_000002.txt
│   └── ...
└── detected_video.mp4         # Video đã annotate
```

## Tối ưu hóa Performance

### 1. **Giảm resolution**
```python
# Thêm vào detect_video() method
results = self.model.predict(
    frame,
    conf=self.conf_threshold,
    imgsz=416,  # Giảm từ 640 (default) xuống 416
    classes=[0]
)
```

### 2. **Sử dụng ONNX export**
```python
# Export sang ONNX để tăng tốc
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx')

# Load ONNX model
detector = PersonDetector(model_path='yolov8n.onnx')
```

### 3. **Batch processing**
Nếu không cần real-time, có thể xử lý theo batch:
```python
results = self.model.predict(
    source='video.mp4',
    conf=0.25,
    classes=[0],
    stream=True,  # Streaming mode tiết kiệm RAM
    vid_stride=2  # Skip frames (xử lý mỗi 2 frame)
)
```

## Xử lý lỗi thường gặp

### 1. **Model không tải được**
```bash
# Download lại model
yolo download model=yolov8n.pt
```

### 2. **Video không mở được**
```python
# Kiểm tra codec
import cv2
cap = cv2.VideoCapture('video.mp4')
print(f"Codec: {int(cap.get(cv2.CAP_PROP_FOURCC))}")
```

### 3. **Out of Memory**
- Giảm `imgsz` xuống 416 hoặc 320
- Sử dụng YOLOv8n thay vì YOLOv8s/m/l
- Bật `stream=True` trong predict

## Đánh giá chất lượng Detection

### Manual Inspection
```python
# Hiển thị một số frames ngẫu nhiên để kiểm tra
detector.detect_video(video_source='video.mp4', show=True)
```

### Với Ground Truth (nếu có)
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
metrics = model.val(data='coco.yaml', split='val')
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## Tích hợp với Bước 2 (Tracking)

Output text files từ bước này sẽ được sử dụng làm input cho Single-Camera Tracking:

```
step1_detection.py → labels/*.txt → step2_tracking.py
```

Format YOLO giúp dễ dàng load và track:
```python
# Đọc detections
with open('labels/frame_000001.txt', 'r') as f:
    for line in f:
        cls, x, y, w, h, conf = map(float, line.strip().split())
        # Convert về pixel coordinates và feed vào tracker
```

## Monitoring & Logging

Thêm logging chi tiết hơn:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection.log'),
        logging.StreamHandler()
    ]
)

# Trong code
logging.info(f"Frame {frame_count}: {num_persons} persons detected")
```

## Next Steps

Sau khi hoàn thành Bước 1:

1. ✅ **Kiểm tra output**: Xem video annotate, đảm bảo detection chính xác
2. ✅ **Đánh giá performance**: FPS có đủ nhanh? (target: >15 FPS)
3. ✅ **Tune parameters**: Điều chỉnh confidence threshold nếu cần
4. ➡️ **Chuyển sang Bước 2**: Single-Camera Tracking với BoT-SORT

## Tham khảo

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLO Format Specification](https://docs.ultralytics.com/datasets/detect/)
- [COCO Dataset Classes](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)
