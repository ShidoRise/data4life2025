"""
Visualize Re-ID features bằng t-SNE hoặc UMAP
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Re-ID features")
    
    parser.add_argument("--features", type=str, required=True,
                        help="File track_features.pkl")
    parser.add_argument("--method", type=str, default="tsne",
                        choices=["tsne", "umap", "pca"])
    parser.add_argument("--output", type=str, default="features_vis.png")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load features
    print(f"[INFO] Loading features from {args.features}")
    with open(args.features, 'rb') as f:
        track_features = pickle.load(f)
    
    # Extract features và labels
    features = []
    labels = []
    
    for track_id, data in track_features.items():
        features.append(data['features'])
        labels.append(track_id)
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"[INFO] Loaded {len(features)} features, dim={features.shape[1]}")
    
    # Dimensionality reduction
    if args.method == "tsne":
        from sklearn.manifold import TSNE
        print("[INFO] Running t-SNE...")
        reducer = TSNE(n_components=2, random_state=42)
        embedded = reducer.fit_transform(features)
    
    elif args.method == "umap":
        try:
            import umap
            print("[INFO] Running UMAP...")
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedded = reducer.fit_transform(features)
        except ImportError:
            print("[ERROR] UMAP not installed. pip install umap-learn")
            return
    
    elif args.method == "pca":
        from sklearn.decomposition import PCA
        print("[INFO] Running PCA...")
        reducer = PCA(n_components=2)
        embedded = reducer.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], 
                         c=labels, cmap='tab20', s=100, alpha=0.7)
    
    # Annotate track IDs
    for i, label in enumerate(labels):
        plt.annotate(f"T{label}", (embedded[i, 0], embedded[i, 1]),
                    fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter, label='Track ID')
    plt.title(f'Re-ID Features Visualization ({args.method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    
    # Save
    output_path = Path(args.output)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] Saved visualization to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
