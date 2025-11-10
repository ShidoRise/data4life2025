"""
BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG Re-ID TỪ TRACKS
=============================================

Mục tiêu:
- Crop người từ tracks (dùng kết quả Step 2)
- Trích xuất đặc trưng Re-ID bằng OSNet
- Lưu features cho Inter-Camera Association (Step 4)
- (Optional) Chuẩn bị dữ liệu để train OSNet với FastReID

Input:
- Video gốc
- Tracks từ Step 2 (MOTChallenge format)

Output:
- Crops của mỗi track (images)
- Features vector (numpy .npy hoặc .pkl)
- Track features summary (JSON)

Flow:
1. Đọc tracks từ MOTChallenge file
2. Crop bbox từ video theo track ID
3. Trích xuất features bằng OSNet pretrained
4. Lưu features + metadata
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Re-ID features from tracks")
    
    parser.add_argument("--source", type=str, required=True,
                        help="Video gốc")
    parser.add_argument("--tracks", type=str, required=True,
                        help="File tracks MOTChallenge (.txt)")
    
    # Re-ID model
    parser.add_argument("--reid-model", type=str,
                        default="osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
                        help="OSNet model weights")
    parser.add_argument("--reid-size", type=int, nargs=2, default=[256, 128],
                        help="Input size cho ReID [height, width]")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="runs/step3_reid_features",
                        help="Thư mục output")
    parser.add_argument("--save-crops", action="store_true",
                        help="Lưu crops của mỗi track")
    parser.add_argument("--max-crops-per-track", type=int, default=10,
                        help="Số crops tối đa mỗi track (lấy sample đều)")
    
    parser.add_argument("--device", type=str, default="",
                        help="Device: 'cpu', '0', 'cuda'")
    
    return parser.parse_args()


def load_mot_tracks(mot_file):
    """
    Load tracks từ MOTChallenge format
    
    Returns:
        dict: {frame_id: [(track_id, x, y, w, h, conf), ...]}
    """
    tracks_by_frame = defaultdict(list)
    
    with open(mot_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            
            tracks_by_frame[frame_id].append({
                'track_id': track_id,
                'bbox': [x, y, w, h],
                'conf': conf
            })
    
    return tracks_by_frame


def extract_reid_model(reid_weights_path, device):
    """
    Load OSNet model cho feature extraction
    
    Args:
        reid_weights_path: Path to .pth weights file
        device: torch.device
    
    Returns:
        model: OSNet model
    """
    if not reid_weights_path or not Path(reid_weights_path).exists():
        print(f"[WARNING] Weights file not found: {reid_weights_path}")
        return None
    
    print(f"[INFO] Loading OSNet weights: {reid_weights_path}")
    
    try:
        import torchreid
        
        # Load checkpoint
        checkpoint = torch.load(reid_weights_path, map_location='cpu')
        
        # Extract state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        print(f"[INFO] Checkpoint has {len(state_dict)} keys")
        
        # Remove 'module.' prefix from DataParallel models
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # Build OSNet x1.0 model for feature extraction
        # This will have the backbone but not the classifier
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=751,  # Market1501 classes
            loss='softmax',
            pretrained=False
        )
        
        print(f"[INFO] OSNet model has {len(model.state_dict())} keys")
        
        # Load weights with strict=False to ignore classifier layers
        # Your weights have full model (backbone + classifier)
        # But we only need backbone for feature extraction
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        # Check if backbone loaded successfully
        backbone_keys = [k for k in new_state_dict.keys() if not k.startswith('classifier')]
        backbone_loaded = [k for k in backbone_keys if k not in missing_keys]
        
        print(f"[INFO] Loaded {len(backbone_loaded)}/{len(backbone_keys)} backbone parameters")
        print(f"[INFO] Ignored {len(unexpected_keys)} classifier parameters (not needed for features)")
        
        if len(backbone_loaded) < len(backbone_keys) * 0.9:
            print("[WARNING] Less than 90% of backbone loaded - features may not be good")
        else:
            print("[INFO] ✓ OSNet backbone loaded successfully!")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"[ERROR] Failed to load OSNet: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_features_from_crops(crops, reid_model, device, reid_size):
    """
    Trích xuất features từ list crops
    
    Args:
        crops: List of BGR images
        reid_model: OSNet model or None
        device: torch.device
        reid_size: (height, width)
    
    Returns:
        np.array: Features [N, feature_dim]
    """
    if len(crops) == 0:
        return np.array([])
    
    # Standard PyTorch model preprocessing
    processed = []
    for crop in crops:
        # Resize
        resized = cv2.resize(crop, (reid_size[1], reid_size[0]))
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        # To tensor [C, H, W]
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        processed.append(tensor)
    
    # Stack batch
    batch = torch.stack(processed).to(device)
    
    # Extract features
    if reid_model is not None:
        with torch.no_grad():
            features = reid_model(batch)
            if isinstance(features, tuple):
                features = features[0]  # Take first output
            features = features.cpu().numpy()
    else:
        # Fallback: dùng ResNet50 pretrained từ torchvision
        print("[INFO] Using ResNet50 as fallback feature extractor")
        import torchvision.models as models
        
        resnet = models.resnet50(weights='IMAGENET1K_V1').to(device)
        resnet.eval()
        
        # Remove classification layer
        feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
        
        with torch.no_grad():
            features = feature_extractor(batch)
            features = features.squeeze().cpu().numpy()
            
            # Handle single sample case
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
    
    return features


def run_extraction(args):
    """Main extraction pipeline"""
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_crops:
        crops_dir = output_dir / 'crops'
        crops_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG Re-ID")
    print("=" * 70)
    print(f"Video:         {args.source}")
    print(f"Tracks file:   {args.tracks}")
    print(f"ReID model:    {args.reid_model}")
    print(f"Output:        {output_dir}")
    print(f"Save crops:    {args.save_crops}")
    print("-" * 70)
    
    # Device
    if args.device:
        device_str = args.device
    else:
        device_str = '0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(f'cuda:{device_str}' if device_str.isdigit() else device_str)
    print(f"[INFO] Device: {device}")
    
    # Load ReID model
    print("[INFO] Loading ReID model...")
    reid_model = extract_reid_model(args.reid_model, device)
    
    if reid_model is not None:
        # Move to device and set to eval mode
        reid_model = reid_model.to(device)
        reid_model.eval()
        print("[INFO] ✓ ReID model ready for feature extraction")
    else:
        print("[WARNING] Using fallback feature extractor (ResNet50)")
    
    # Load tracks
    print("[INFO] Loading tracks...")
    tracks_by_frame = load_mot_tracks(args.tracks)
    print(f"[INFO] Loaded {len(tracks_by_frame)} frames with tracks")
    
    # Get unique track IDs
    all_track_ids = set()
    for frame_tracks in tracks_by_frame.values():
        for track in frame_tracks:
            all_track_ids.add(track['track_id'])
    
    print(f"[INFO] Found {len(all_track_ids)} unique tracks")
    
    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {args.source}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Collect frame indices for each track (lightweight)
    track_frame_bboxes = defaultdict(list)  # {track_id: [(frame_idx, bbox), ...]}
    
    print("\n[INFO] Indexing tracks...")
    for frame_idx, frame_tracks in tracks_by_frame.items():
        for track in frame_tracks:
            track_id = track['track_id']
            track_frame_bboxes[track_id].append((frame_idx, track['bbox']))
    
    print(f"[INFO] Indexed {len(track_frame_bboxes)} tracks")
    
    # Process each track one by one (memory efficient)
    print("\n[INFO] Extracting Re-ID features...")
    
    track_features = {}
    track_metadata = {}
    
    for track_id in tqdm(sorted(track_frame_bboxes.keys()), desc="Extract features"):
        frame_bbox_list = track_frame_bboxes[track_id]
        
        # Sample frames nếu quá nhiều
        if len(frame_bbox_list) > args.max_crops_per_track:
            indices = np.linspace(0, len(frame_bbox_list) - 1, args.max_crops_per_track, dtype=int)
            frame_bbox_list = [frame_bbox_list[i] for i in indices]
        
        # Extract crops from video for this track
        crops = []
        frame_indices = []
        
        for frame_idx, bbox in frame_bbox_list:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # Clamp to boundaries
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2].copy()  # Copy to avoid reference issues
            
            if crop.size == 0:
                continue
            
            crops.append(crop)
            frame_indices.append(frame_idx)
            
            # Save crop nếu cần
            if args.save_crops:
                track_dir = crops_dir / f"track_{track_id:04d}"
                track_dir.mkdir(exist_ok=True)
                crop_file = track_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(crop_file), crop)
        
        # Extract features for this track
        if len(crops) > 0:
            features = extract_features_from_crops(
                crops, reid_model, device, args.reid_size
            )
            
            if len(features) > 0:
                # Average pooling
                avg_feature = features.mean(axis=0)
                
                track_features[track_id] = {
                    'features': avg_feature,
                    'all_features': features,
                    'frame_indices': frame_indices
                }
                
                track_metadata[track_id] = {
                    'num_frames': len(crops),
                    'frame_indices': frame_indices,
                    'feature_dim': avg_feature.shape[0]
                }
        
        # Clear crops from memory
        del crops
    
    cap.release()
    
    # Lưu features
    print("\n[INFO] Saving features...")
    
    features_file = output_dir / 'track_features.pkl'
    with open(features_file, 'wb') as f:
        pickle.dump(track_features, f)
    print(f"[INFO] Saved to: {features_file}")
    
    # Lưu metadata
    metadata_file = output_dir / 'track_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(track_metadata, f, indent=2)
    print(f"[INFO] Metadata: {metadata_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total tracks:       {len(track_features)}")
    print(f"Feature dimension:  {list(track_features.values())[0]['features'].shape[0] if track_features else 0}")
    print(f"Output directory:   {output_dir}")
    if args.save_crops:
        print(f"Crops saved:        {crops_dir}")
    print("=" * 70)


def main():
    args = parse_args()
    
    # Validate
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Video not found: {args.source}")
    if not Path(args.tracks).exists():
        raise FileNotFoundError(f"Tracks file not found: {args.tracks}")
    
    run_extraction(args)


if __name__ == "__main__":
    main()
