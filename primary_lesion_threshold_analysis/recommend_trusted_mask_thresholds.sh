#!/bin/bash

# ---------------------- Configuration ----------------------
# Set CUDA device
CUDA_VISIBLE_DEVICES="0"
DEVICE="cuda:${CUDA_VISIBLE_DEVICES}"

# Model checkpoint path
CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/train_dataset_4/20260113_170943/dino_unet_train_dataset_4_epoch_50.pth"

# Labeled primary-lesion validation/test dataset
DATASET_NAME="Malignant_ultrasound"
IMAGE_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_4/test/images/"
MASK_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_4/test/masks/"

# Output directory for threshold analysis
OUTPUT_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/threshold_analysis_outputs/Malignant_ultrasound"

# Inference parameters
IMG_SIZE=224
BATCH_SIZE=4
NUM_WORKERS=4
DINO_PRETRAINED="True"
USE_DILATION="False"
SAVE_ORIG_SIZE="True"
PRED_THRESHOLD="0.5"
MIN_REGION_AREA=32

# Quality-aligned threshold recommendation
QUALITY_RULE="dice_only"
QUALITY_DICE_MIN="0.70"
QUALITY_HD95_MAX=""
QUALITY_ECE_MAX=""
LOWER_QUANTILE="0.10"
UPPER_QUANTILE="0.90"
STRICT_PAIR_CHECK="True"

# ---------------------- Execution ----------------------
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file does not exist: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory does not exist: $IMAGE_DIR"
    exit 1
fi

if [ ! -d "$MASK_DIR" ]; then
    echo "Error: Mask directory does not exist: $MASK_DIR"
    exit 1
fi

if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/recommend_trusted_mask_thresholds.py"

CMD=(
    python -u "$PYTHON_SCRIPT"
    --checkpoint "$CHECKPOINT_PATH"
    --image_dir "$IMAGE_DIR"
    --mask_dir "$MASK_DIR"
    --dataset_name "$DATASET_NAME"
    --output_dir "$OUTPUT_DIR"
    --img_size "$IMG_SIZE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --device "$DEVICE"
    --dino_pretrained "$DINO_PRETRAINED"
    --use_dilation "$USE_DILATION"
    --save_orig_size "$SAVE_ORIG_SIZE"
    --pred_threshold "$PRED_THRESHOLD"
    --min_region_area "$MIN_REGION_AREA"
    --quality_rule "$QUALITY_RULE"
    --quality_dice_min "$QUALITY_DICE_MIN"
    --lower_quantile "$LOWER_QUANTILE"
    --upper_quantile "$UPPER_QUANTILE"
    --strict_pair_check "$STRICT_PAIR_CHECK"
)

if [ -n "$QUALITY_HD95_MAX" ]; then
    CMD+=(--quality_hd95_max "$QUALITY_HD95_MAX")
fi

if [ -n "$QUALITY_ECE_MAX" ]; then
    CMD+=(--quality_ece_max "$QUALITY_ECE_MAX")
fi

echo "Running threshold analysis for: $DATASET_NAME"
echo "Image dir: $IMAGE_DIR"
echo "Mask dir: $MASK_DIR"
echo "Output dir: $OUTPUT_DIR"

"${CMD[@]}"
