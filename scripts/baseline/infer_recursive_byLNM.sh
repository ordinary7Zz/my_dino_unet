#!/bin/bash

# ---------------------- Configuration ----------------------
# Set CUDA device
CUDA_VISIBLE_DEVICES="0"
DEVICE="cuda:${CUDA_VISIBLE_DEVICES}"

# Model checkpoint path
CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/train_Lymph_Node_Metastasis/Lymph_Node_Metastasis/20260521_211141/dino_unet_Lymph_Node_Metastasis_epoch_40.pth"

# Input image root directory (recursive)
INPUT_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/"

# Prediction results save path
OUTPUT_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions_ByLNM"

# Inference parameters
IMG_SIZE=224
BATCH_SIZE=4
NUM_WORKERS=4
DINO_PRETRAINED="True"
USE_DILATION="False"
SAVE_ORIG_SIZE="True"
OUTPUT_TYPE="binary"   # binary | prob
THRESHOLD="0.5"

# ---------------------- Execution ----------------------
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file does not exist: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python -u infer_recursive.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --img_size "$IMG_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --dino_pretrained "$DINO_PRETRAINED" \
    --use_dilation "$USE_DILATION" \
    --save_orig_size "$SAVE_ORIG_SIZE" \
    --output_type "$OUTPUT_TYPE" \
    --threshold "$THRESHOLD"
