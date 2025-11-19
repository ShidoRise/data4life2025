"""
Helper script to test MTMC with single camera by splitting video
"""

import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split video to simulate multi-camera setup")
    
    parser.add_argument("--source", type=str, required=True,
                        help="Input video file")
    parser.add_argument("--num-splits", type=int, default=2,
                        help="Number of splits (cameras)")
    parser.add_argument("--output-dir", type=str, default="runs/simulated_cameras",
                        help="Output directory for split videos")
    
    return parser.parse_args()


def split_video(source, num_splits, output_dir):
    """Split video into N equal parts"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video duration using ffprobe
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        source
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
    except Exception as e:
        print(f"[ERROR] Failed to get video duration: {e}")
        print("[INFO] Trying alternative method...")
        
        # Alternative: use opencv
        import cv2
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
    
    print(f"[INFO] Video duration: {duration:.2f} seconds")
    
    segment_duration = duration / num_splits
    split_files = []
    
    for i in range(num_splits):
        start_time = i * segment_duration
        output_file = output_dir / f"cam{i}.mp4"
        
        cmd = [
            'ffmpeg', '-y',
            '-i', source,
            '-ss', str(start_time),
            '-t', str(segment_duration),
            '-c', 'copy',
            str(output_file)
        ]
        
        print(f"[INFO] Creating cam{i}.mp4 (from {start_time:.1f}s to {start_time + segment_duration:.1f}s)...")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            split_files.append(output_file)
            print(f"[INFO] ✓ Saved: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to split video: {e}")
            return []
    
    return split_files


def main():
    args = parse_args()
    
    print("=" * 70)
    print("SIMULATE MULTI-CAMERA SETUP")
    print("=" * 70)
    print(f"Source video:  {args.source}")
    print(f"Num cameras:   {args.num_splits}")
    print(f"Output dir:    {args.output_dir}")
    print("-" * 70)
    
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Video not found: {args.source}")
    
    split_files = split_video(args.source, args.num_splits, args.output_dir)
    
    if split_files:
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Run tracking on each 'camera':")
        
        for i, video_file in enumerate(split_files):
            print(f"\npython step2_tracking.py \\")
            print(f"    --source {video_file} \\")
            print(f"    --output-dir runs/cam{i}_tracking")
        
        print("\n2. Extract features for each camera:")
        
        for i, video_file in enumerate(split_files):
            print(f"\npython step3_reid_extraction.py \\")
            print(f"    --source {video_file} \\")
            print(f"    --tracks runs/cam{i}_tracking/tracks/cam{i}.txt \\")
            print(f"    --output-dir runs/cam{i}_features \\")
            print(f"    --device 0")
        
        print("\n3. Run inter-camera association:")
        print(f"\npython step4_inter_camera_association.py \\")
        print(f"    --features", end='')
        for i in range(args.num_splits):
            print(f" runs/cam{i}_features/track_features.pkl", end='')
        print(" \\")
        print(f"    --camera-names", end='')
        for i in range(args.num_splits):
            print(f" cam{i}", end='')
        print(" \\")
        print(f"    --method hungarian \\")
        print(f"    --similarity-threshold 0.6 \\")
        print(f"    --output-dir runs/step4_mtmc \\")
        print(f"    --visualize")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
