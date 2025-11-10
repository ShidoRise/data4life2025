"""
BƯỚC 4: INTER-CAMERA ASSOCIATION (MTMC)
========================================

Mục tiêu:
- Liên kết tracks từ nhiều cameras dựa trên Re-ID features
- Gán Global ID cho cùng một người xuất hiện trên nhiều cameras
- Sử dụng cosine similarity + Hungarian algorithm hoặc clustering

Input:
- Track features từ Step 3 (từ nhiều cameras)
- Metadata của tracks

Output:
- Global ID mapping: {camera_id: {local_track_id: global_id}}
- Similarity matrix
- Association results (JSON)

Flow:
1. Load features từ tất cả cameras
2. Tính cosine similarity giữa tracks từ các cameras khác nhau
3. Áp dụng Hungarian algorithm hoặc clustering (DBSCAN)
4. Gán Global ID và export results
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine, cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inter-camera association for MTMC")
    
    parser.add_argument("--features", type=str, nargs='+', required=True,
                        help="Paths to track_features.pkl files from different cameras")
    parser.add_argument("--camera-names", type=str, nargs='+',
                        help="Camera names (optional, will use cam0, cam1, ... if not provided)")
    
    # Association method
    parser.add_argument("--method", type=str, default="hungarian",
                        choices=["hungarian", "clustering"],
                        help="Association method")
    
    # Hungarian parameters
    parser.add_argument("--similarity-threshold", type=float, default=0.6,
                        help="Minimum cosine similarity for matching (0-1)")
    
    # Clustering parameters
    parser.add_argument("--eps", type=float, default=0.5,
                        help="DBSCAN eps parameter (max distance for clustering)")
    parser.add_argument("--min-samples", type=int, default=2,
                        help="DBSCAN min_samples parameter")
    
    # Time constraints (optional)
    parser.add_argument("--time-window", type=int, default=None,
                        help="Maximum time difference (frames) for association")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="runs/step4_mtmc",
                        help="Output directory")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize similarity matrix")
    
    return parser.parse_args()


def load_camera_features(features_file, camera_name):
    """
    Load track features from a single camera
    
    Returns:
        dict: {track_id: {'features': array, 'metadata': dict}}
    """
    with open(features_file, 'rb') as f:
        track_features = pickle.load(f)
    
    # Add camera info to each track
    camera_tracks = {}
    for track_id, data in track_features.items():
        camera_tracks[track_id] = {
            'camera': camera_name,
            'features': data['features'],
            'frame_indices': data.get('frame_indices', []),
            'feature_dim': data['features'].shape[0]
        }
    
    return camera_tracks


def compute_similarity_matrix(tracks_cam1, tracks_cam2, metric='cosine'):
    """
    Compute similarity matrix between tracks from two cameras
    
    Args:
        tracks_cam1: dict of tracks from camera 1
        tracks_cam2: dict of tracks from camera 2
        metric: 'cosine' or 'euclidean'
    
    Returns:
        similarity_matrix: numpy array [N1, N2]
        track_ids_1: list of track IDs from camera 1
        track_ids_2: list of track IDs from camera 2
    """
    track_ids_1 = sorted(tracks_cam1.keys())
    track_ids_2 = sorted(tracks_cam2.keys())
    
    # Extract features
    features_1 = np.array([tracks_cam1[tid]['features'] for tid in track_ids_1])
    features_2 = np.array([tracks_cam2[tid]['features'] for tid in track_ids_2])
    
    # Normalize features for cosine similarity
    if metric == 'cosine':
        features_1 = features_1 / (np.linalg.norm(features_1, axis=1, keepdims=True) + 1e-8)
        features_2 = features_2 / (np.linalg.norm(features_2, axis=1, keepdims=True) + 1e-8)
        # Cosine similarity = dot product of normalized vectors
        similarity_matrix = np.dot(features_1, features_2.T)
    else:  # euclidean
        # Convert distance to similarity
        distance_matrix = cdist(features_1, features_2, metric='euclidean')
        similarity_matrix = 1.0 / (1.0 + distance_matrix)
    
    return similarity_matrix, track_ids_1, track_ids_2


def hungarian_association(similarity_matrix, track_ids_1, track_ids_2, 
                         cam1_name, cam2_name, threshold=0.6):
    """
    Associate tracks using Hungarian algorithm
    
    Returns:
        matches: list of (cam1_track_id, cam2_track_id, similarity)
    """
    # Convert similarity to cost (Hungarian minimizes cost)
    cost_matrix = 1.0 - similarity_matrix
    
    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    for i, j in zip(row_ind, col_ind):
        similarity = similarity_matrix[i, j]
        if similarity >= threshold:
            matches.append({
                'cam1': cam1_name,
                'track1': track_ids_1[i],
                'cam2': cam2_name,
                'track2': track_ids_2[j],
                'similarity': float(similarity)
            })
    
    return matches


def clustering_association(all_tracks, eps=0.5, min_samples=2):
    """
    Associate tracks across all cameras using DBSCAN clustering
    
    Args:
        all_tracks: dict {camera_name: {track_id: track_data}}
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
    
    Returns:
        clusters: dict {cluster_id: [(camera, track_id), ...]}
    """
    # Collect all features with camera/track info
    features_list = []
    track_info = []  # [(camera, track_id), ...]
    
    for camera, tracks in all_tracks.items():
        for track_id, data in tracks.items():
            features_list.append(data['features'])
            track_info.append((camera, track_id))
    
    features_array = np.array(features_list)
    
    # Normalize features
    features_array = features_array / (np.linalg.norm(features_array, axis=1, keepdims=True) + 1e-8)
    
    # DBSCAN clustering (using cosine distance)
    # Note: eps is distance threshold, smaller eps = tighter clusters
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(features_array)
    
    # Group tracks by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:  # Ignore noise
            camera, track_id = track_info[idx]
            clusters[label].append({
                'camera': camera,
                'track_id': track_id,
                'features': features_list[idx]
            })
    
    return dict(clusters)


def assign_global_ids(matches=None, clusters=None, all_tracks=None):
    """
    Assign global IDs based on matches or clusters
    
    Returns:
        global_id_map: {camera: {local_track_id: global_id}}
    """
    global_id_map = defaultdict(dict)
    next_global_id = 1
    
    if matches is not None:
        # Hungarian method: pairwise matches
        # Build connected components
        from collections import defaultdict
        graph = defaultdict(set)
        
        for match in matches:
            key1 = (match['cam1'], match['track1'])
            key2 = (match['cam2'], match['track2'])
            graph[key1].add(key2)
            graph[key2].add(key1)
        
        # Find connected components (same person across cameras)
        visited = set()
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                
                # Assign same global ID to all tracks in this component
                for camera, track_id in component:
                    global_id_map[camera][track_id] = next_global_id
                
                next_global_id += 1
        
        # Assign unique IDs to unmatched tracks
        for camera, tracks in all_tracks.items():
            for track_id in tracks.keys():
                if track_id not in global_id_map[camera]:
                    global_id_map[camera][track_id] = next_global_id
                    next_global_id += 1
    
    elif clusters is not None:
        # Clustering method: directly from clusters
        for cluster_id, cluster_tracks in clusters.items():
            for track in cluster_tracks:
                camera = track['camera']
                track_id = track['track_id']
                global_id_map[camera][track_id] = cluster_id + 1  # Start from 1
        
        # Assign unique IDs to tracks not in any cluster
        assigned_tracks = {(t['camera'], t['track_id']) 
                          for cluster in clusters.values() 
                          for t in cluster}
        
        max_cluster_id = max(clusters.keys()) if clusters else 0
        next_id = max_cluster_id + 2
        
        for camera, tracks in all_tracks.items():
            for track_id in tracks.keys():
                if (camera, track_id) not in assigned_tracks:
                    global_id_map[camera][track_id] = next_id
                    next_id += 1
    
    return dict(global_id_map)


def visualize_similarity_matrix(similarity_matrix, track_ids_1, track_ids_2,
                                cam1_name, cam2_name, output_file):
    """Visualize similarity matrix as heatmap"""
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(similarity_matrix, 
                xticklabels=track_ids_2,
                yticklabels=track_ids_1,
                cmap='viridis',
                vmin=0, vmax=1,
                cbar_kws={'label': 'Cosine Similarity'})
    
    plt.xlabel(f'{cam2_name} Track IDs')
    plt.ylabel(f'{cam1_name} Track IDs')
    plt.title(f'Track Similarity: {cam1_name} vs {cam2_name}')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"[INFO] Similarity matrix saved to: {output_file}")


def run_association(args):
    """Main association pipeline"""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("BƯỚC 4: INTER-CAMERA ASSOCIATION (MTMC)")
    print("=" * 70)
    print(f"Method:        {args.method}")
    print(f"Cameras:       {len(args.features)}")
    print(f"Output:        {output_dir}")
    print("-" * 70)
    
    # Camera names
    if args.camera_names and len(args.camera_names) == len(args.features):
        camera_names = args.camera_names
    else:
        camera_names = [f"cam{i}" for i in range(len(args.features))]
    
    # Load features from all cameras
    print("\n[INFO] Loading track features from all cameras...")
    all_tracks = {}
    
    for features_file, camera_name in zip(args.features, camera_names):
        if not Path(features_file).exists():
            print(f"[WARNING] File not found: {features_file}")
            continue
        
        tracks = load_camera_features(features_file, camera_name)
        all_tracks[camera_name] = tracks
        print(f"[INFO] {camera_name}: {len(tracks)} tracks")
    
    if len(all_tracks) < 2:
        raise ValueError("Need at least 2 cameras for MTMC!")
    
    total_tracks = sum(len(tracks) for tracks in all_tracks.values())
    print(f"[INFO] Total tracks: {total_tracks}")
    
    # Association
    if args.method == "hungarian":
        print(f"\n[INFO] Running Hungarian algorithm (threshold={args.similarity_threshold})...")
        
        all_matches = []
        camera_pairs = [(c1, c2) for i, c1 in enumerate(camera_names) 
                       for c2 in camera_names[i+1:]]
        
        for cam1, cam2 in tqdm(camera_pairs, desc="Computing matches"):
            # Compute similarity
            sim_matrix, ids1, ids2 = compute_similarity_matrix(
                all_tracks[cam1], all_tracks[cam2], metric='cosine'
            )
            
            # Find matches
            matches = hungarian_association(
                sim_matrix, ids1, ids2, cam1, cam2, 
                threshold=args.similarity_threshold
            )
            
            all_matches.extend(matches)
            
            print(f"[INFO] {cam1} <-> {cam2}: {len(matches)} matches")
            
            # Visualize if requested
            if args.visualize:
                vis_file = output_dir / f"similarity_{cam1}_vs_{cam2}.png"
                visualize_similarity_matrix(sim_matrix, ids1, ids2, 
                                          cam1, cam2, vis_file)
        
        # Assign global IDs
        print("\n[INFO] Assigning global IDs...")
        global_id_map = assign_global_ids(matches=all_matches, 
                                         all_tracks=all_tracks)
        
        # Save matches
        matches_file = output_dir / 'pairwise_matches.json'
        with open(matches_file, 'w') as f:
            json.dump(all_matches, f, indent=2)
        print(f"[INFO] Matches saved to: {matches_file}")
    
    else:  # clustering
        print(f"\n[INFO] Running DBSCAN clustering (eps={args.eps}, min_samples={args.min_samples})...")
        
        clusters = clustering_association(all_tracks, 
                                        eps=args.eps, 
                                        min_samples=args.min_samples)
        
        print(f"[INFO] Found {len(clusters)} clusters")
        
        # Assign global IDs
        global_id_map = assign_global_ids(clusters=clusters, 
                                         all_tracks=all_tracks)
        
        # Save clusters
        clusters_file = output_dir / 'clusters.json'
        clusters_serializable = {
            str(k): [{'camera': t['camera'], 'track_id': int(t['track_id'])} 
                    for t in v]
            for k, v in clusters.items()
        }
        with open(clusters_file, 'w') as f:
            json.dump(clusters_serializable, f, indent=2)
        print(f"[INFO] Clusters saved to: {clusters_file}")
    
    # Save global ID mapping
    global_id_file = output_dir / 'global_id_mapping.json'
    # Convert keys to strings for JSON
    global_id_map_serializable = {
        cam: {str(tid): gid for tid, gid in tracks.items()}
        for cam, tracks in global_id_map.items()
    }
    with open(global_id_file, 'w') as f:
        json.dump(global_id_map_serializable, f, indent=2)
    print(f"[INFO] Global ID mapping saved to: {global_id_file}")
    
    # Statistics
    unique_global_ids = set()
    for tracks in global_id_map.values():
        unique_global_ids.update(tracks.values())
    
    print("\n" + "=" * 70)
    print("ASSOCIATION SUMMARY")
    print("=" * 70)
    print(f"Total local tracks:     {total_tracks}")
    print(f"Unique global IDs:      {len(unique_global_ids)}")
    print(f"Reduction:              {total_tracks - len(unique_global_ids)} tracks merged")
    
    for camera, tracks in global_id_map.items():
        print(f"  {camera}: {len(tracks)} tracks")
    
    print("=" * 70)


def main():
    args = parse_args()
    run_association(args)


if __name__ == "__main__":
    main()
