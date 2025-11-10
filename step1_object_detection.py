"""
BƯỚC 1: OBJECT DETECTION VỚI YOLOv8
===================================

Pipeline MTMC - Phần 1: Phát hiện đối tượng (người) trong video

Mục tiêu:
---------
- Sử dụng YOLOv8n (lightweight) để phát hiện người (class 0) trong video
- Lưu kết quả detection dưới dạng text files (YOLO format)
- Trực quan hóa kết quả bằng bounding boxes trên video
- Đánh giá hiệu năng (FPS, mAP nếu có ground truth)

Yêu cầu:
--------
- ultralytics (đã cài đặt)
- YOLOv8n model (yolov8n.pt) - đã có trong workspace
- Video đầu vào hoặc webcam

Output:
-------
- Video đã annotate với bounding boxes
- Text files chứa detection results (YOLO format: class x_center y_center width height confidence)
- Thống kê: số người phát hiện được, FPS, thời gian xử lý

"""

import os
import time
import cv2
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class PersonDetector:
    """
    Class để phát hiện người trong video sử dụng YOLOv8
    """
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25, iou_threshold=0.45):
        """
        Khởi tạo detector
        
        Args:
            model_path (str): Đường dẫn đến YOLO model
            conf_threshold (float): Ngưỡng confidence để giữ detection (0-1)
            iou_threshold (float): Ngưỡng IoU cho NMS
        """
        print(f"[INFO] Đang tải YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = 0  # COCO dataset: class 0 = person
        
        print(f"[INFO] Model đã tải thành công!")
        print(f"[INFO] Confidence threshold: {conf_threshold}")
        print(f"[INFO] IoU threshold: {iou_threshold}")
    
    def detect_video(self, video_source, output_dir='runs/detect', 
                     save_txt=True, save_video=True, show=False):
        """
        Phát hiện người trong video
        
        Args:
            video_source (str): Đường dẫn video hoặc '0' cho webcam
            output_dir (str): Thư mục lưu kết quả
            save_txt (bool): Lưu detection dưới dạng text files
            save_video (bool): Lưu video đã annotate
            show (bool): Hiển thị video real-time
        
        Returns:
            dict: Thống kê kết quả detection
        """
        # Tạo thư mục output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_txt:
            labels_path = output_path / 'labels'
            labels_path.mkdir(exist_ok=True)
        
        # Mở video
        cap = cv2.VideoCapture(video_source if video_source != '0' else 0)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {video_source}")
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n[INFO] Thông tin video:")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - FPS: {fps}")
        print(f"  - Total frames: {total_frames}")
        
        # Khởi tạo video writer nếu cần
        video_writer = None
        if save_video:
            output_video = output_path / f'detected_{Path(video_source).name}'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video), fourcc, fps, (width, height)
            )
            print(f"[INFO] Video output sẽ được lưu tại: {output_video}")
        
        # Thống kê
        frame_count = 0
        total_detections = 0
        processing_times = []
        
        print("\n[INFO] Bắt đầu xử lý video...")
        print("-" * 70)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                start_time = time.time()
                
                # Chạy detection
                results = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=[self.person_class_id],  # Chỉ phát hiện người
                    verbose=False
                )
                
                process_time = time.time() - start_time
                processing_times.append(process_time)
                
                # Lấy detections
                detections = results[0].boxes
                num_persons = len(detections)
                total_detections += num_persons
                
                # Vẽ bounding boxes
                annotated_frame = results[0].plot()
                
                # Thêm thông tin lên frame
                info_text = f"Frame: {frame_count}/{total_frames} | Persons: {num_persons} | FPS: {1/process_time:.1f}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Lưu detections vào text file (YOLO format)
                if save_txt and num_persons > 0:
                    txt_file = labels_path / f'frame_{frame_count:06d}.txt'
                    with open(txt_file, 'w') as f:
                        for box in detections:
                            # Chuyển sang YOLO format: class x_center y_center width height confidence
                            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                            conf = box.conf[0].cpu().numpy()
                            
                            # Normalize coordinates
                            x_center = ((xyxy[0] + xyxy[2]) / 2) / width
                            y_center = ((xyxy[1] + xyxy[3]) / 2) / height
                            w = (xyxy[2] - xyxy[0]) / width
                            h = (xyxy[3] - xyxy[1]) / height
                            
                            f.write(f"{self.person_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
                
                # Lưu video
                if save_video:
                    video_writer.write(annotated_frame)
                
                # Hiển thị
                if show:
                    cv2.imshow('Person Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[INFO] Người dùng dừng xử lý.")
                        break
                
                # In tiến độ
                if frame_count % 30 == 0:
                    avg_fps = 1 / np.mean(processing_times[-30:])
                    print(f"Frame {frame_count}/{total_frames} | Avg FPS: {avg_fps:.1f} | Persons detected: {num_persons}")
        
        finally:
            # Giải phóng tài nguyên
            cap.release()
            if video_writer:
                video_writer.release()
            if show:
                cv2.destroyAllWindows()
        
        # Tính toán thống kê
        avg_processing_time = np.mean(processing_times)
        avg_fps = 1 / avg_processing_time
        
        stats = {
            'total_frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0,
            'avg_processing_time': avg_processing_time,
            'avg_fps': avg_fps,
            'output_dir': str(output_path)
        }
        
        # In báo cáo
        print("\n" + "=" * 70)
        print("KẾT QUẢ DETECTION")
        print("=" * 70)
        print(f"Tổng số frames xử lý:          {stats['total_frames']}")
        print(f"Tổng số người phát hiện:       {stats['total_detections']}")
        print(f"Trung bình người/frame:        {stats['avg_detections_per_frame']:.2f}")
        print(f"Thời gian xử lý TB/frame:      {stats['avg_processing_time']*1000:.2f} ms")
        print(f"FPS trung bình:                {stats['avg_fps']:.2f}")
        print(f"Kết quả lưu tại:               {stats['output_dir']}")
        print("=" * 70)
        
        return stats
    
    def detect_images(self, image_dir, output_dir='runs/detect_images', 
                      save_txt=True, save_images=True):
        """
        Phát hiện người trong thư mục ảnh
        
        Args:
            image_dir (str): Thư mục chứa ảnh
            output_dir (str): Thư mục lưu kết quả
            save_txt (bool): Lưu detection dưới dạng text files
            save_images (bool): Lưu ảnh đã annotate
        
        Returns:
            dict: Thống kê kết quả
        """
        image_path = Path(image_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_txt:
            labels_path = output_path / 'labels'
            labels_path.mkdir(exist_ok=True)
        
        if save_images:
            images_path = output_path / 'images'
            images_path.mkdir(exist_ok=True)
        
        # Lấy danh sách ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in image_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"[INFO] Tìm thấy {len(image_files)} ảnh")
        
        total_detections = 0
        
        for idx, img_file in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Xử lý: {img_file.name}")
            
            # Đọc ảnh
            img = cv2.imread(str(img_file))
            height, width = img.shape[:2]
            
            # Detect
            results = self.model.predict(
                img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[self.person_class_id],
                verbose=False
            )
            
            detections = results[0].boxes
            num_persons = len(detections)
            total_detections += num_persons
            
            # Lưu text
            if save_txt and num_persons > 0:
                txt_file = labels_path / f'{img_file.stem}.txt'
                with open(txt_file, 'w') as f:
                    for box in detections:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        x_center = ((xyxy[0] + xyxy[2]) / 2) / width
                        y_center = ((xyxy[1] + xyxy[3]) / 2) / height
                        w = (xyxy[2] - xyxy[0]) / width
                        h = (xyxy[3] - xyxy[1]) / height
                        
                        f.write(f"{self.person_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
            
            # Lưu ảnh annotate
            if save_images:
                annotated = results[0].plot()
                output_img = images_path / img_file.name
                cv2.imwrite(str(output_img), annotated)
            
            print(f"  -> Phát hiện {num_persons} người")
        
        print(f"\n[INFO] Hoàn thành! Tổng: {total_detections} người trong {len(image_files)} ảnh")
        
        return {
            'total_images': len(image_files),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(image_files) if image_files else 0
        }


def main():
    """
    Hàm chính để demo detection
    """
    print("=" * 70)
    print("BƯỚC 1: OBJECT DETECTION - PHÁT HIỆN NGƯỜI TRONG VIDEO")
    print("=" * 70)
    
    # Khởi tạo detector
    detector = PersonDetector(
        model_path='yolov8n.pt',
        conf_threshold=0.25,  # Giảm xuống nếu muốn phát hiện nhiều hơn
        iou_threshold=0.45
    )
    
    # Ví dụ 1: Detect từ video file
    # Thay đổi đường dẫn video của bạn ở đây
    video_source = 'people-walking.mp4'  # hoặc đường dẫn video khác
    
    if not os.path.exists(video_source):
        print(f"\n[WARNING] Không tìm thấy video: {video_source}")
        print("[INFO] Vui lòng cung cấp video hoặc sử dụng webcam (source='0')")
        return
    
    # Chạy detection
    stats = detector.detect_video(
        video_source=video_source,
        output_dir='runs/step1_detection',
        save_txt=True,      # Lưu text files cho bước tracking sau
        save_video=True,    # Lưu video annotate
        show=False          # Đổi thành True nếu muốn xem real-time
    )
    
    print("\n[SUCCESS] Bước 1 hoàn thành!")
    print("[NEXT] Các text files detection sẽ được sử dụng cho Bước 2: Single-Camera Tracking")
    
    # Ví dụ 2: Detect từ webcam (uncomment để sử dụng)
    # detector.detect_video(video_source='0', output_dir='runs/webcam_detection', show=True)
    
    # Ví dụ 3: Detect từ thư mục ảnh (uncomment để sử dụng)
    # detector.detect_images(image_dir='path/to/images', output_dir='runs/image_detection')


if __name__ == '__main__':
    main()
