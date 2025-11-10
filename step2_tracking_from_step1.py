"""
BƯỚC 2 (VARIANT): TRACKING VỚI DETECTIONS CÓ SẴN TỪ STEP 1
===========================================================

Script này load detections từ Step 1 (labels/*.txt) thay vì chạy YOLO lại.
Tiết kiệm thời gian và đảm bảo nhất quán giữa detection và tracking.

Input:
- Video gốc
- Thư mục labels/ từ Step 1 (YOLO format)

Output:
- Video tracked với ID
- MOTChallenge format tracks
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tracking with Pre-computed Detections")
    
    parser.add_argument("--source", type=str, required=True,
                        help="Video gốc (cùng video dùng ở Step 1)")
    parser.add_argument("--detections", type=str, required=True,
                        help="Thư mục chứa detection labels từ Step 1")
    
    # Tracker
    parser.add_argument("--tracker-type", type=str, default="botsort",
                        choices=["botsort", "bytetrack"])
    parser.add_argument("--with-reid", action="store_true")
    parser.add_argument("--reid-model", type=str,
                        default="osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth")
    parser.add_argument("--tracker-config", type=str, default="botsort_config.yaml")
    
    # Output
    parser.add_argument("--project", type=str, default="runs/step2_from_step1")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--show", action="store_true")
    
    parser.add_argument("--device", type=str, default="")
    
    return parser.parse_args()


def load_detections_for_frame(labels_dir, frame_idx, img_width, img_height):
    """
    Load detections từ file YOLO format
    
    Returns:
        np.array: [x1, y1, x2, y2, conf, cls] format cho BoxMOT
    """
    label_file = Path(labels_dir) / f"frame_{frame_idx:06d}.txt"
    
    if not label_file.exists():
        return np.empty((0, 6))
    
    detections = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            cls, x_center, y_center, w, h, conf = map(float, parts)
            
            # Convert từ YOLO format (normalized) sang pixel coords
            x_center *= img_width
            y_center *= img_height
            w *= img_width
            h *= img_height
            
            # Convert sang [x1, y1, x2, y2]
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            detections.append([x1, y1, x2, y2, conf, cls])
    
    return np.array(detections) if detections else np.empty((0, 6))


def run_tracking_from_detections(args):
    """Tracking với detections có sẵn"""
    from boxmot import create_tracker
    
    # Setup
    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = output_dir / 'tracks'
    tracks_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("BƯỚC 2: TRACKING VỚI DETECTIONS TỪ STEP 1")
    print("=" * 70)
    print(f"Video:         {args.source}")
    print(f"Detections:    {args.detections}")
    print(f"Tracker:       {args.tracker_type}")
    print(f"With ReID:     {args.with_reid}")
    print(f"Output:        {output_dir}")
    print("-" * 70)
    
    # Device setup
    if args.device:
        device_str = args.device
    else:
        device_str = '0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(f'cuda:{device_str}' if device_str.isdigit() else device_str)
    print(f"[INFO] Device: {device}")
    
    # Tạo tracker
    tracker_config = Path(args.tracker_config) if Path(args.tracker_config).exists() else None
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
        print(f"[INFO] Tracker created: {args.tracker_type}")
    except Exception as e:
        print(f"[ERROR] {e}")
        print("[INFO] Retrying with default config...")
        tracker = create_tracker(
            tracker_type=args.tracker_type,
            tracker_config=None,
            reid_weights=reid_weights,
            device=device,
            half=False,
            per_class=False
        )
    
    # Mở video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {args.source}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n[INFO] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Output video
    video_writer = None
    if args.save:
        output_video = output_dir / f"tracked_{Path(args.source).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # MOTChallenge file
    mot_file = None
    if args.save_txt:
        mot_file = open(tracks_dir / f"{Path(args.source).stem}.txt", 'w')
    
    print("\n[INFO] Tracking...")
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
            start_time = time.time()
            
            # Load detections từ file
            det = load_detections_for_frame(args.detections, frame_idx, width, height)
            
            # Update tracker
            if len(det) > 0:
                tracks = tracker.update(det, frame)
            else:
                tracks = np.empty((0, 8))
            
            process_time = time.time() - start_time
            processing_times.append(process_time)
            
            # Vẽ
            annotated_frame = frame.copy()
            
            if len(tracks) > 0:
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    conf = track[5]
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{track_id} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if mot_file:
                        w = x2 - x1
                        h = y2 - y1
                        mot_file.write(f"{frame_idx},{track_id},{x1},{y1},{w},{h},{conf:.4f},-1,-1,-1\n")
                
                total_tracks = max(total_tracks, int(tracks[:, 4].max()))
            
            # Info
            info = f"Frame: {frame_idx}/{total_frames} | Tracks: {len(tracks)} | FPS: {1/process_time:.1f}"
            cv2.putText(annotated_frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if video_writer:
                video_writer.write(annotated_frame)
            
            if args.show:
                cv2.imshow('Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_idx % 30 == 0:
                avg_fps = 1 / np.mean(processing_times[-30:])
                print(f"Frame {frame_idx}/{total_frames} | Tracks: {len(tracks)} | Avg FPS: {avg_fps:.1f}")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if mot_file:
            mot_file.close()
        if args.show:
            cv2.destroyAllWindows()
    
    avg_fps = 1 / np.mean(processing_times) if processing_times else 0
    
    print("\n" + "=" * 70)
    print("TRACKING SUMMARY")
    print("=" * 70)
    print(f"Frames:        {frame_idx}")
    print(f"Total tracks:  {total_tracks}")
    print(f"Avg FPS:       {avg_fps:.2f}")
    print(f"Output:        {output_dir}")
    print("=" * 70)


def main():
    args = parse_args()
    
    # Validate
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Video not found: {args.source}")
    if not Path(args.detections).exists():
        raise FileNotFoundError(f"Detections dir not found: {args.detections}")
    if args.with_reid and not Path(args.reid_model).exists():
        raise FileNotFoundError(f"ReID model not found: {args.reid_model}")
    
    run_tracking_from_detections(args)


if __name__ == "__main__":
    main()
