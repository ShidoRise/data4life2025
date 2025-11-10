"""
Helper script: Chuẩn bị dữ liệu từ crops sang FastReID format
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert crops to FastReID dataset format")
    
    parser.add_argument("--crops-dir", type=str, required=True,
                        help="Thư mục crops từ step3_reid_extraction")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Thư mục output cho FastReID")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Tỷ lệ train/test split")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    crops_dir = Path(args.crops_dir)
    output_dir = Path(args.output_dir)
    
    # Tạo thư mục
    train_dir = output_dir / 'train'
    query_dir = output_dir / 'query'
    gallery_dir = output_dir / 'gallery'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Converting crops from {crops_dir} to FastReID format...")
    
    # Lấy tất cả track folders
    track_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir()])
    
    print(f"[INFO] Found {len(track_dirs)} tracks")
    
    for track_dir in track_dirs:
        track_id = track_dir.name  # track_0001, track_0002, ...
        
        # Lấy tất cả crops
        crops = sorted(list(track_dir.glob('*.jpg')))
        
        if len(crops) < 2:
            print(f"[WARNING] Track {track_id} has < 2 crops, skipping")
            continue
        
        # Split train/test
        num_train = int(len(crops) * args.train_ratio)
        
        train_crops = crops[:num_train]
        test_crops = crops[num_train:]
        
        # Copy train
        train_track_dir = train_dir / track_id
        train_track_dir.mkdir(exist_ok=True)
        
        for crop in train_crops:
            shutil.copy(crop, train_track_dir / crop.name)
        
        # Copy test (query + gallery)
        if len(test_crops) > 0:
            query_track_dir = query_dir / track_id
            gallery_track_dir = gallery_dir / track_id
            
            query_track_dir.mkdir(exist_ok=True)
            gallery_track_dir.mkdir(exist_ok=True)
            
            # First crop → query, rest → gallery
            shutil.copy(test_crops[0], query_track_dir / test_crops[0].name)
            
            for crop in test_crops[1:]:
                shutil.copy(crop, gallery_track_dir / crop.name)
        
        print(f"[INFO] {track_id}: {len(train_crops)} train, {len(test_crops)} test")
    
    print(f"\n[SUCCESS] Dataset created at: {output_dir}")
    print(f"  Train: {train_dir}")
    print(f"  Query: {query_dir}")
    print(f"  Gallery: {gallery_dir}")


if __name__ == "__main__":
    main()
