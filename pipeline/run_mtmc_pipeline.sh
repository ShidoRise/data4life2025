#!/bin/bash
# Batch processing for multi-camera MTMC pipeline

# Configuration
DEVICE=0
MAX_CROPS=10
SIMILARITY_THRESHOLD=0.6

# Camera videos (thay đổi paths này)
declare -a CAMERAS=(
    "camera1:/path/to/camera1.mp4"
    "camera2:/path/to/camera2.mp4"
    "camera3:/path/to/camera3.mp4"
)

echo "========================================="
echo "MTMC PIPELINE - BATCH PROCESSING"
echo "========================================="
echo "Cameras: ${#CAMERAS[@]}"
echo ""

# Arrays to store paths
FEATURE_FILES=()
CAMERA_NAMES=()

# Process each camera
for cam_data in "${CAMERAS[@]}"; do
    # Split camera name and path
    IFS=':' read -r cam_name cam_path <<< "$cam_data"
    
    echo "========================================"
    echo "Processing: $cam_name"
    echo "Video: $cam_path"
    echo "========================================"
    
    # Step 2: Tracking
    echo "[Step 2] Running tracking..."
    python step2_tracking.py \
        --source "$cam_path" \
        --output-dir "runs/${cam_name}_tracking" \
        --device $DEVICE
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Tracking failed for $cam_name"
        exit 1
    fi
    
    # Find the tracks file (BoxMOT creates exp/ subdirs)
    TRACKS_FILE=$(find "runs/${cam_name}_tracking" -name "*.txt" -path "*/tracks/*" | head -n 1)
    
    if [ -z "$TRACKS_FILE" ]; then
        echo "[ERROR] Tracks file not found for $cam_name"
        exit 1
    fi
    
    echo "[INFO] Tracks file: $TRACKS_FILE"
    
    # Step 3: Feature extraction
    echo "[Step 3] Extracting Re-ID features..."
    python step3_reid_extraction.py \
        --source "$cam_path" \
        --tracks "$TRACKS_FILE" \
        --output-dir "runs/${cam_name}_features" \
        --device $DEVICE \
        --max-crops-per-track $MAX_CROPS
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Feature extraction failed for $cam_name"
        exit 1
    fi
    
    # Store paths for Step 4
    FEATURE_FILES+=("runs/${cam_name}_features/track_features.pkl")
    CAMERA_NAMES+=("$cam_name")
    
    echo "[INFO] ✓ $cam_name completed"
    echo ""
done

# Step 4: Inter-camera association
echo "========================================="
echo "[Step 4] Running inter-camera association"
echo "========================================="

# Build command arguments
FEATURES_ARG=""
NAMES_ARG=""

for feature_file in "${FEATURE_FILES[@]}"; do
    FEATURES_ARG="$FEATURES_ARG $feature_file"
done

for cam_name in "${CAMERA_NAMES[@]}"; do
    NAMES_ARG="$NAMES_ARG $cam_name"
done

# Run association
python step4_inter_camera_association.py \
    --features $FEATURES_ARG \
    --camera-names $NAMES_ARG \
    --method hungarian \
    --similarity-threshold $SIMILARITY_THRESHOLD \
    --output-dir runs/mtmc_results \
    --visualize

if [ $? -ne 0 ]; then
    echo "[ERROR] Inter-camera association failed"
    exit 1
fi

echo ""
echo "========================================="
echo "PIPELINE COMPLETED!"
echo "========================================="
echo "Results saved to: runs/mtmc_results/"
echo ""
echo "Output files:"
echo "  - global_id_mapping.json"
echo "  - pairwise_matches.json"
echo "  - similarity_*.png (heatmaps)"
echo ""
echo "To view global ID mapping:"
echo "  cat runs/mtmc_results/global_id_mapping.json"
echo ""
