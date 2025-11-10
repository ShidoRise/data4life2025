# Hướng dẫn triển khai pipeline MTMC (Multi-Target Multi-Camera Tracking)

Một pipeline MTMC điển hình gồm bốn bước chính: (1) **Object Detection**
trên từng camera để lấy bounding-box người, (2) **Single-Camera Tracking
(SCT)** để liên kết các phát hiện qua các khung hình trong cùng camera,
(3) **Trích xuất đặc trưng Re-ID** từ mỗi tracklet, và (4) **Liên kết
liên camera** (Inter-Camera Association) dựa trên độ tương đồng đặc
trưng. Ở đây chúng ta sử dụng thư viện **BoxMOT** (đã tích hợp sẵn
tracker BoT-SORT) để gộp các thành phần này thành một hệ thống
"plug-and-play". Đối với phần Re-ID, ta chọn **OSNet** -- một kiến trúc
rất nhẹ (≈2.2M tham số, \~0.98 GFLOPs) nhưng đạt hiệu suất SOTA trên
nhiều bộ dữ liệu Re-ID. Mô hình OSNet sẽ được huấn luyện bằng
**FastReID** với các kỹ thuật "Bag of Tricks". Ngoài ra, để tăng độ
chính xác mà không cần huấn luyện thêm, ta áp dụng **Pose2ID** (CVPR
2025) như một lớp hậu xử lý (training-free) và phương pháp thích ứng
miền thời gian thực **PaTTA-ID** để khắc phục domain shift trên thiết bị
yếu. Dưới đây là các bước triển khai chi tiết:

## Bước 1: Cài đặt môi trường và Phát hiện đối tượng (YOLOv8/v9)

1.  **Cài đặt các thư viện cần thiết**: Tạo môi trường Python, cài đặt
    BoxMOT (`pip install boxmot` hoặc clone GitHub), FastReID, và
    Ultralytics YOLO (`pip install ultralytics`). Nếu không có GPU mạnh,
    chọn phiên bản nhẹ của YOLO (YOLOv8n).

2.  **Mô hình YOLO**: Sử dụng YOLOv8 hoặc YOLOv9 để phát hiện người
    trong từng khung hình. BoxMOT hỗ trợ các detector YOLOv8/v9 và các
    mô hình khác. Ví dụ:

    ``` bash
    yolo detect model=yolov8n.pt source=/path/to/videos --save-txt
    ```

## Bước 2: Theo dõi Đơn Camera (Single-Camera Tracking)

1.  **Chọn tracker**: Sử dụng **BoT-SORT** trong BoxMOT. BoT-SORT kết
    hợp motion (Kalman Filter) và appearance (Re-ID).

2.  **Chạy theo dõi**:

    ``` bash
    boxmot eval --yolo-model yolov8n.pt --reid-model osnet_x0_25_market1501.pt --tracking-method botsort --source /path/to/camera_streams
    ```

    Có thể chỉnh cấu hình `botsort.yaml` trong BoxMOT để tinh chỉnh
    tracker.

## Bước 3: Huấn luyện mô hình Re-ID OSNet (FastReID + Bag of Tricks)

1.  **Chuẩn bị dữ liệu**: Sử dụng Market-1501 hoặc DukeMTMC.

2.  **FastReID và BoT**: FastReID hỗ trợ OSNet và tích hợp các "Bag of
    Tricks" (warmup LR, random erasing, BNNeck, label smoothing).

3.  **Huấn luyện**:

    ``` bash
    python tools/train.py --config_file configs/OSNet/osnet_x1_0_market1501.yaml
    ```

4.  **Áp dụng Pose2ID** (tùy chọn): Dùng Pose2ID (CVPR 2025) để "Feature
    Centralization" và tăng độ chính xác mà không cần huấn luyện lại.

## Bước 4: Liên kết Đa-Camera (Inter-Camera Association)

Liên kết tracklet giữa các camera dựa trên độ tương đồng cosine giữa
vector đặc trưng. Có thể dùng thuật toán Hungarian hoặc clustering
(DBSCAN). BoxMOT xử lý tự động nếu tracker hỗ trợ Re-ID như BoT-SORT.

## Bước 5: Tăng cường Re-ID với Pose2ID (CVPR 2025)

Pose2ID sinh thêm ảnh với nhiều pose khác nhau, thực hiện "Feature
Centralization" để có vector đặc trưng ổn định hơn. Tham khảo GitHub
chính thức: [yuanc3/Pose2ID](https://github.com/yuanc3/Pose2ID).

## Bước 6: Thích ứng Miền (Domain Adaptation)

Sử dụng **PaTTA-ID (ArXiv 2024)** để thực hiện Test-Time Adaptation cho
phần cứng yếu. Nó điều chỉnh trọng số mô hình trong khi chạy, giảm hiệu
ứng "domain shift". Tham khảo [PaTTA-ID
OpenReview](https://openreview.net/pdf?id=69cdb8c924d0f9db3cfb052df4df3c966c02546d).

## Khuyến nghị cho phần cứng yếu

-   **YOLO nhẹ**: YOLOv8n hoặc YOLOv8s.
-   **OSNet nhỏ**: OSNet-x0.25 hoặc x0.5.
-   **ONNX/TensorRT**: Xuất model OSNet sang ONNX để tăng tốc.
-   **Giảm batch size** trong FastReID.
-   **Giới hạn Pose2ID**: Chạy trên subset dữ liệu nếu thiếu GPU.

## Tài nguyên

-   [BoxMOT GitHub](https://github.com/mikel-brostrom/boxmot)
-   [BoT-SORT GitHub](https://github.com/NirAharon/BoT-SORT)
-   [FastReID GitHub](https://github.com/JDAI-CV/fast-reid)
-   [Pose2ID (CVPR 2025)](https://github.com/yuanc3/Pose2ID)
-   [PaTTA-ID
    Paper](https://openreview.net/pdf?id=69cdb8c924d0f9db3cfb052df4df3c966c02546d)
