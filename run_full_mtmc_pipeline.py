"""
Full MTMC Pipeline Runner
Runs Steps 2, 3, 4 automatically for multiple cameras
"""

import argparse
import subprocess
import json
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full MTMC pipeline for multiple cameras")
    
    parser.add_argument("--videos", type=str, nargs='+', required=True,
                        help="Paths to camera videos")
    parser.add_argument("--camera-names", type=str, nargs='+',
                        help="Camera names (optional)")
    
    # Processing options
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device")
    parser.add_argument("--max-crops", type=int, default=10,
                        help="Max crops per track for ReID")
    
    # Association options
    parser.add_argument("--method", type=str, default="hungarian",
                        choices=["hungarian", "clustering"],
                        help="Association method")
    parser.add_argument("--similarity-threshold", type=float, default=0.6,
                        help="Similarity threshold (Hungarian)")
    parser.add_argument("--eps", type=float, default=0.5,
                        help="DBSCAN eps (Clustering)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Base output directory")
    parser.add_argument("--skip-tracking", action="store_true",
                        help="Skip Step 2 if already done")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip Step 3 if already done")
    
    return parser.parse_args()


def run_command(cmd, description):
    """Run shell command and handle errors"""
    print(f"\n[INFO] {description}")
    print(f"[CMD] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Command failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    return True


def find_tracks_file(tracking_dir):
    """Find the MOTChallenge tracks file in tracking output"""
    tracks_dir = Path(tracking_dir)
    
    # BoxMOT creates tracks in exp/tracks/ or similar
    for tracks_file in tracks_dir.rglob("*.txt"):
        if "tracks" in str(tracks_file):
            return tracks_file
    
    return None


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Camera names
    if args.camera_names and len(args.camera_names) == len(args.videos):
        camera_names = args.camera_names
    else:
        camera_names = [f"camera{i+1}" for i in range(len(args.videos))]
    
    print("=" * 70)
    print("MTMC PIPELINE - AUTOMATIC RUNNER")
    print("=" * 70)
    print(f"Cameras:       {len(args.videos)}")
    print(f"Device:        {args.device}")
    print(f"Method:        {args.method}")
    print(f"Output:        {output_dir}")
    print("-" * 70)
    
    # Store paths for Step 4
    feature_files = []
    
    # Process each camera
    for video_path, cam_name in zip(args.videos, camera_names):
        print("\n" + "=" * 70)
        print(f"PROCESSING: {cam_name}")
        print("=" * 70)
        print(f"Video: {video_path}")
        
        if not Path(video_path).exists():
            print(f"[ERROR] Video not found: {video_path}")
            continue
        
        tracking_dir = output_dir / f"{cam_name}_tracking"
        features_dir = output_dir / f"{cam_name}_features"
        
        # Step 2: Tracking
        if not args.skip_tracking:
            cmd = [
                "python", "step2_tracking.py",
                "--source", video_path,
                "--output-dir", str(tracking_dir),
                "--device", args.device
            ]
            
            if not run_command(cmd, f"[Step 2] Running tracking for {cam_name}"):
                print(f"[ERROR] Tracking failed for {cam_name}")
                continue
        else:
            print(f"[INFO] Skipping tracking (already done)")
        
        # Find tracks file
        tracks_file = find_tracks_file(tracking_dir)
        if not tracks_file:
            print(f"[ERROR] Tracks file not found in {tracking_dir}")
            continue
        
        print(f"[INFO] Tracks file: {tracks_file}")
        
        # Step 3: Feature extraction
        if not args.skip_features:
            cmd = [
                "python", "step3_reid_extraction.py",
                "--source", video_path,
                "--tracks", str(tracks_file),
                "--output-dir", str(features_dir),
                "--device", args.device,
                "--max-crops-per-track", str(args.max_crops)
            ]
            
            if not run_command(cmd, f"[Step 3] Extracting features for {cam_name}"):
                print(f"[ERROR] Feature extraction failed for {cam_name}")
                continue
        else:
            print(f"[INFO] Skipping feature extraction (already done)")
        
        # Check features file exists
        features_file = features_dir / "track_features.pkl"
        if not features_file.exists():
            print(f"[ERROR] Features file not found: {features_file}")
            continue
        
        feature_files.append(str(features_file))
        print(f"[INFO] ✓ {cam_name} completed")
    
    # Step 4: Inter-camera association
    if len(feature_files) < 2:
        print("\n[ERROR] Need at least 2 cameras with features for MTMC!")
        return
    
    print("\n" + "=" * 70)
    print("[STEP 4] INTER-CAMERA ASSOCIATION")
    print("=" * 70)
    
    mtmc_output = output_dir / "mtmc_results"
    
    cmd = [
        "python", "step4_inter_camera_association.py",
        "--features"
    ] + feature_files + [
        "--camera-names"
    ] + camera_names + [
        "--method", args.method,
        "--output-dir", str(mtmc_output),
        "--visualize"
    ]
    
    if args.method == "hungarian":
        cmd += ["--similarity-threshold", str(args.similarity_threshold)]
    else:
        cmd += ["--eps", str(args.eps)]
    
    if not run_command(cmd, "Running inter-camera association"):
        print("[ERROR] Association failed!")
        return
    
    # Display results
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED!")
    print("=" * 70)
    
    global_id_file = mtmc_output / "global_id_mapping.json"
    if global_id_file.exists():
        with open(global_id_file, 'r') as f:
            global_ids = json.load(f)
        
        print("\nGlobal ID Statistics:")
        for cam, tracks in global_ids.items():
            print(f"  {cam}: {len(tracks)} tracks")
        
        unique_ids = set()
        for tracks in global_ids.values():
            unique_ids.update(tracks.values())
        
        total_tracks = sum(len(tracks) for tracks in global_ids.values())
        print(f"\nTotal local tracks: {total_tracks}")
        print(f"Unique global IDs:  {len(unique_ids)}")
        print(f"Reduction:          {total_tracks - len(unique_ids)} tracks merged")
    
    print(f"\nResults saved to: {mtmc_output}")
    print("\nOutput files:")
    print("  - global_id_mapping.json")
    print("  - pairwise_matches.json (or clusters.json)")
    print("  - similarity_*.png (heatmaps)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
