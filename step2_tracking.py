"""
BƯỚC 2: SINGLE-CAMERA TRACKING (BoT-SORT via BoxMOT)
=====================================================

Mục tiêu:
- Dùng BoxMOT (BoT-SORT) để theo dõi người trên từng camera
- Tích hợp trực tiếp với YOLOv8 detector
- Lưu video kết quả và file track theo chuẩn MOTChallenge

Yêu cầu:
- Môi trường conda "boxmot" có sẵn boxmot, ultralytics, torch, opencv
- File model YOLO: yolov8n.pt (có sẵn trong workspace)
- File model ReID (OSNet): osnet_x1_0_market_*.pth (có sẵn trong workspace)

Output:
- Video đã vẽ track ID
- Tracks ở định dạng MOTChallenge (.txt)
- Per-frame track info

BoxMOT API:
- create_tracker(tracker_type, tracker_config, reid_weights, device, half, per_class)
- tracker.update(dets, img) -> tracks
"""

import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-Camera Tracking with BoT-SORT (BoxMOT)")

    # Nguồn vào
    parser.add_argument("--source", type=str, default="people-walking.mp4",
                        help="Đường dẫn video, thư mục, hoặc '0' cho webcam")

    # Detector
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="Đường dẫn YOLO model (.pt/.onnx)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước ảnh đầu vào cho YOLO")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold cho detector")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold cho NMS")
    parser.add_argument("--vid-stride", type=int, default=1,
                        help="Bỏ qua frame, ví dụ 2 = xử lý mỗi 2 frame")

    # Tracker/ReID
    parser.add_argument("--tracker-type", type=str, default="botsort",
                        choices=["botsort", "bytetrack"],
                        help="Loại tracker sử dụng")
    parser.add_argument("--with-reid", action="store_true",
                        help="Bật ReID (khuyến nghị cho BoT-SORT)")
    parser.add_argument("--reid-model", type=str,
                        default="osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
                        help="Đường dẫn model ReID (OSNet .pth)")
    parser.add_argument("--tracker-config", type=str, default="botsort_config.yaml",
                        help="File cấu hình cho BoT-SORT")

    # Lưu/hiển thị
    parser.add_argument("--project", type=str, default="runs/step2_tracking",
                        help="Thư mục gốc lưu kết quả")
    parser.add_argument("--name", type=str, default="exp",
                        help="Tên phiên chạy (subfolder)")
    parser.add_argument("--save", action="store_true",
                        help="Lưu video kết quả")
    parser.add_argument("--save-txt", action="store_true",
                        help="Lưu tracks theo chuẩn MOTChallenge")
    parser.add_argument("--show", action="store_true",
                        help="Hiển thị video real-time")

    # Thiết bị
    parser.add_argument("--device", type=str, default="",
                        help="Thiết bị: 'cpu' hoặc '0' (GPU0), để trống = auto")

    return parser.parse_args()


def validate_files(args):
    # YOLO model
    if not Path(args.yolo_model).exists():
        raise FileNotFoundError(f"Không tìm thấy YOLO model: {args.yolo_model}")

    # ReID model if with-reid enabled
    if args.with_reid and not Path(args.reid_model).exists():
        raise FileNotFoundError(f"ReID model not found: {args.reid_model}")

    # Tracker config
    if args.tracker_type == "botsort" and args.tracker_config and not Path(args.tracker_config).exists():
        print(f"[WARNING] Tracker config not found: {args.tracker_config}. Will use BoxMOT defaults.")

    # Source input
    if args.source != "0" and not Path(args.source).exists():
        raise FileNotFoundError(f"Source not found: {args.source}")


def run_tracking(args):
    """Chạy tracking với BoxMOT API"""
    from ultralytics import YOLO
    from boxmot import create_tracker, get_tracker_config
    
    # Tạo thư mục output
    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracks_dir = output_dir / 'tracks'
    tracks_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("STEP 2: SINGLE-CAMERA TRACKING - BoT-SORT (BoxMOT)")
    print("=" * 70)
    print(f"Source:        {args.source}")
    print(f"YOLO model:    {args.yolo_model}")
    print(f"Tracker:       {args.tracker_type}")
    print(f"With ReID:     {args.with_reid}")
    if args.with_reid:
        print(f"ReID model:    {args.reid_model}")
    print(f"Tracker config: {args.tracker_config if Path(args.tracker_config).exists() else 'default'}")
    print(f"Output dir:    {output_dir}")
    print(f"Save video:    {args.save} | Save txt: {args.save_txt} | Show: {args.show}")
    print("-" * 70)

    # Load YOLO detector
    print("\n[INFO] Loading YOLO model...")
    model = YOLO(args.yolo_model)
    
    # Setup device
    if args.device:
        device_str = args.device
    else:
        device_str = '0' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(f'cuda:{device_str}' if device_str.isdigit() else device_str)
    print(f"[INFO] Using device: {device}")

    # Tạo tracker
    print(f"[INFO] Creating {args.tracker_type} tracker...")
    tracker_config = None
    if Path(args.tracker_config).exists():
        try:
            tracker_config = Path(args.tracker_config)
            print(f"[INFO] Using config: {tracker_config}")
        except Exception as e:
            print(f"[WARNING] Cannot load config {args.tracker_config}: {e}")
            print("[INFO] Using default tracker config")
            tracker_config = None
    
    reid_weights = Path(args.reid_model) if args.with_reid else None
    
    try:
        tracker = create_tracker(
            tracker_type=args.tracker_type,
            tracker_config=tracker_config,
            reid_weights=reid_weights,
            device=device,
            half=False,
            per_class=False
        )
    except Exception as e:
        print(f"[ERROR] Failed to create tracker with config: {e}")
        print("[INFO] Retrying without config file (using defaults)...")
        tracker = create_tracker(
            tracker_type=args.tracker_type,
            tracker_config=None,
            reid_weights=reid_weights,
            device=device,
            half=False,
            per_class=False
        )
    
    # Mở video
    cap = cv2.VideoCapture(args.source if args.source != '0' else 0)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {args.source}")
    
    # Thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n[INFO] Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Video writer
    video_writer = None
    if args.save:
        output_video = output_dir / f"tracked_{Path(args.source).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        print(f"[INFO] Output video: {output_video}")
    
    # MOTChallenge file
    mot_file = None
    if args.save_txt:
        mot_file = open(tracks_dir / f"{Path(args.source).stem}.txt", 'w')
        print(f"[INFO] MOTChallenge output: {tracks_dir / Path(args.source).stem}.txt")
    
    print("\n[INFO] Starting tracking...")
    print("-" * 70)
    
    frame_idx = 0
    total_tracks = 0
    processing_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip frames if vid_stride > 1
            if frame_idx % args.vid_stride != 0:
                continue
            
            start_time = time.time()
            
            # YOLO detection
            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                classes=[0],  # person only
                verbose=False,
                imgsz=args.imgsz,
                device=device_str  # YOLO dùng string
            )
            
            # Lấy detections
            det = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
            
            # Update tracker
            if len(det) > 0:
                # BoxMOT expects: [x1, y1, x2, y2, conf, cls]
                tracks = tracker.update(det, frame)
            else:
                tracks = np.empty((0, 8))  # No detections
            
            process_time = time.time() - start_time
            processing_times.append(process_time)
            
            # Vẽ tracks lên frame
            annotated_frame = frame.copy()
            
            if len(tracks) > 0:
                for track in tracks:
                    # tracks format: [x1, y1, x2, y2, track_id, conf, cls, idx]
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    conf = track[5]
                    
                    # Vẽ bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Vẽ ID
                    label = f"ID:{track_id} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Ghi MOTChallenge format
                    if mot_file:
                        # Format: frame, id, x, y, w, h, conf, -1, -1, -1
                        w = x2 - x1
                        h = y2 - y1
                        mot_file.write(f"{frame_idx},{track_id},{x1},{y1},{w},{h},{conf:.4f},-1,-1,-1\n")
                
                total_tracks = max(total_tracks, int(tracks[:, 4].max()))
            
            # Info text
            info = f"Frame: {frame_idx}/{total_frames} | Tracks: {len(tracks)} | FPS: {1/process_time:.1f}"
            cv2.putText(annotated_frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save/show
            if video_writer:
                video_writer.write(annotated_frame)
            
            if args.show:
                cv2.imshow('Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] User stopped.")
                    break
            
            # Progress
            if frame_idx % 30 == 0:
                avg_fps = 1 / np.mean(processing_times[-30:])
                print(f"Frame {frame_idx}/{total_frames} | Active tracks: {len(tracks)} | Avg FPS: {avg_fps:.1f}")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if mot_file:
            mot_file.close()
        if args.show:
            cv2.destroyAllWindows()
    
    # Stats
    avg_fps = 1 / np.mean(processing_times) if processing_times else 0
    
    print("\n" + "=" * 70)
    print("TRACKING SUMMARY")
    print("=" * 70)
    print(f"Processed frames:     {frame_idx}")
    print(f"Total unique tracks:  {total_tracks}")
    print(f"Average FPS:          {avg_fps:.2f}")
    print(f"Output directory:     {output_dir}")
    print("=" * 70)
    
    return {
        'frames': frame_idx,
        'tracks': total_tracks,
        'fps': avg_fps,
        'output_dir': str(output_dir)
    }


def main():
    args = parse_args()
    validate_files(args)
    run_tracking(args)


if __name__ == "__main__":
    main()
