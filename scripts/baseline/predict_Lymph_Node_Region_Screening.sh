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
OUTPUT_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions_Lymph_Node_Region_Screening"
CONFIG_PATH="configs/lymph_node_region_screening.yaml"

# Inference parameters
IMG_SIZE=224
BATCH_SIZE=4
NUM_WORKERS=4
DINO_PRETRAINED="True"
USE_DILATION="False"
SAVE_ORIG_SIZE="True"
THRESHOLD="0.5"
SAVE_SUBSET_MANIFEST="True"

INFERENCE_OUTPUT_DIR="${OUTPUT_DIR}/masks"
REPORT_DIR="${OUTPUT_DIR}/reports"
OUTPUT_CSV="${REPORT_DIR}/screening_results.csv"
BINARY_MASK_DIR="${INFERENCE_OUTPUT_DIR}/binary"
PROB_MASK_DIR="${INFERENCE_OUTPUT_DIR}/prob"

# ---------------------- Execution ----------------------
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file does not exist: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file does not exist: $CONFIG_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$INFERENCE_OUTPUT_DIR" "$REPORT_DIR"

python -u infer_recursive.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$INFERENCE_OUTPUT_DIR" \
    --img_size "$IMG_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --dino_pretrained "$DINO_PRETRAINED" \
    --use_dilation "$USE_DILATION" \
    --save_orig_size "$SAVE_ORIG_SIZE" \
    --threshold "$THRESHOLD" \
    --save_binary "True" \
    --save_prob_npy "True"

python -u screen_lymph_node_region.py \
    --image_dir "$INPUT_DIR" \
    --binary_mask_dir "$BINARY_MASK_DIR" \
    --prob_mask_dir "$PROB_MASK_DIR" \
    --output_csv "$OUTPUT_CSV" \
    --config "$CONFIG_PATH" \
    --save_subset_manifest "$SAVE_SUBSET_MANIFEST" \
    --manifest_dir "$REPORT_DIR"
