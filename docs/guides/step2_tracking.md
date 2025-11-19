# BƯỚC 2: SINGLE-CAMERA TRACKING (BoT-SORT via BoxMOT)

Bước này liên kết các bounding boxes qua thời gian trong cùng một camera để tạo ra các track ID ổn định. Ta dùng BoT-SORT (kết hợp motion + ReID) thông qua thư viện BoxMOT.

## Tính năng chính
- Tích hợp trực tiếp YOLOv8 làm detector đầu vào
- Bật/tắt ReID (OSNet) linh hoạt
- Tuỳ chỉnh tham số BoT-SORT qua file `botsort_config.yaml`
- Lưu video kết quả và file track theo chuẩn MOTChallenge

## Cách chạy nhanh

```bash
# Kích hoạt đúng môi trường trước (đã có boxmot & ultralytics)
conda activate boxmot

# Ví dụ chạy theo dõi 1 video có sẵn
python step2_tracking.py \
  --source people-walking.mp4 \
  --yolo-model yolov8n.pt \
  --tracker-type botsort \
  --with-reid \
  --reid-model osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth \
  --tracker-config botsort_config.yaml \
  --save --save-txt --show \
  --project runs/step2_tracking --name exp
```

- `--save`: Lưu video kết quả (đã annotate ID)
- `--save-txt`: Xuất file track chuẩn MOTChallenge (để chấm điểm/đánh giá hoặc dùng downstream)
- `--show`: Xem real-time (tắt nếu chạy headless)

## Tham số quan trọng

- `--conf`, `--iou`: Ảnh hưởng số lượng detections đầu vào tracker
- `--vid-stride`: Bỏ qua frame để tăng FPS (ví dụ 2 = xử lý mỗi 2 frame)
- `--with-reid`: Nên bật cho BoT-SORT để tăng độ ổn định ID
- `--tracker-config`: Điều chỉnh các ngưỡng trong BoT-SORT (track buffer, match thresh, v.v.)

## Cấu trúc Output

```
runs/step2_tracking/exp/
├── video_result.mp4          # Tên thật phụ thuộc vào BoxMOT/Ultralytics
├── tracks/                   # (tuỳ phiên bản BoxMOT) chứa .txt theo MOTChallenge
│   └── video.txt
└── ...
```

Định dạng MOTChallenge (mỗi dòng là 1 bbox/track trong 1 frame):
```
frame, id, x, y, w, h, conf, x, y, z
```
- `frame`: chỉ số frame (1-based)
- `id`: ID của track
- `x, y, w, h`: toạ độ và kích thước bbox theo pixel
- `conf`: độ tin cậy (có thể là score trung bình)

## File cấu hình BoT-SORT (`botsort_config.yaml`)

Các khoá thường dùng (ví dụ):
```
# Ví dụ, có thể khác tuỳ bản BoxMOT
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.7
track_buffer: 30
match_thresh: 0.8
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: true
# ...
```
Bạn có thể tinh chỉnh tuỳ theo video/camera. Nếu `botsort_config.yaml` không tồn tại, script sẽ để BoxMOT dùng mặc định bên trong thư viện.

## Mẹo tối ưu hiệu năng
- Dùng YOLOv8n + giảm `imgsz` về 416/320
- Bật `--vid-stride 2` để bỏ qua bớt khung hình
- Chạy trên GPU nếu có (mặc định BoxMOT/Ultralytics sẽ tự phát hiện). Ép CPU bằng `--device cpu` nếu cần.

## Tích hợp với Bước 1

Nếu bạn đã chạy Bước 1, thường không cần xuất detections ra file riêng vì `step2_tracking.py` sẽ chạy detector trực tiếp. Tuy nhiên, bạn có thể thay detector bằng detections có sẵn nếu sử dụng API/nội bộ BoxMOT nâng cao (ngoài phạm vi script này).

## Đánh giá chất lượng (tuỳ chọn)

Nếu có ground-truth theo MOTChallenge, bạn có thể chấm IDF1/MOTA bằng các tool bên ngoài như `py-motmetrics`. Trong repo này, chúng tôi chỉ cung cấp stub và hướng dẫn; chưa tích hợp đánh giá tự động để giữ cho Bước 2 gọn nhẹ.

## Lỗi thường gặp
- Không import được `boxmot`: chắc chắn đã `conda activate boxmot`
- Không tìm thấy model ReID: kiểm tra lại đường dẫn `--reid-model`
- Video không mở được: kiểm tra codec/đường dẫn; thử `--source 0` để dùng webcam

## Tiếp theo
Khi Single-Camera Tracking đã ổn, ta sẽ chuyển sang Bước 3: Trích xuất đặc trưng ReID/huấn luyện OSNet (hoặc dùng model có sẵn) và chuẩn bị cho liên kết liên camera (Bước 4).
