#!/bin/bash

# ---------------------- Configuration ----------------------
Train_DATASET="dataset_2"
# Set CUDA device
CUDA_VISIBLE_DEVICES="1"
# 根据 CUDA_VISIBLE_DEVICES 自动设置 device，也可以手动指定（如 "cuda:0", "cpu"）
DEVICE="cuda:${CUDA_VISIBLE_DEVICES}"

# Model checkpoint path
CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/Dilation/dino_unet_Dilation_train_${Train_DATASET}_epoch_50.pth"

# Configure multiple test dataset paths
# Test dataset names array
TEST_DATASET_NAMES=(
    "TN3K"
    "DDTI"
    "ThyroidXL"
    "PKTN"
    "TN5K"
    "Cine-Clip"
)

# 测试图像路径数组
TEST_IMAGE_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI/test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/PKTN/test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/images/"
)

# 测试掩码路径数组
TEST_MASK_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI/test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/PKTN/test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/masks/"
)

# Ensure arrays have the same length
if [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_IMAGE_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_MASK_PATHS[@]} ]; then
    echo "Error: Arrays must have the same length"
    exit 1
fi

# Prediction results save path
SAVE_PATH="./predictions/Dilation/new_test_${Train_DATASET}"

# Whether to save prediction results (true/false)
SAVE_RESULTS="false"

# Log directory
LOG_DIR="./logs/test_logs/Dilation/new_test_${Train_DATASET}"

# ---------------------- Execution ----------------------
# Create save directory if it doesn't exist
mkdir -p "$SAVE_PATH"

# Build test image paths arguments
TEST_IMAGE_ARGS=()
for img_path in "${TEST_IMAGE_PATHS[@]}"; do
    if [ -d "$img_path" ]; then
        TEST_IMAGE_ARGS+=("--test_image_paths" "$img_path")
    fi
done

# Build test mask paths arguments
TEST_MASK_ARGS=()
for mask_path in "${TEST_MASK_PATHS[@]}"; do
    if [ -d "$mask_path" ]; then
        TEST_MASK_ARGS+=("--test_gt_paths" "$mask_path")
    fi
done

# Build test dataset names arguments
TEST_NAMES_ARGS=()
for dataset_name in "${TEST_DATASET_NAMES[@]}"; do
    TEST_NAMES_ARGS+=("--test_dataset_names" "$dataset_name")
done

python -u "test_parallel.py" \
    --checkpoint "$CHECKPOINT_PATH" \
    "${TEST_IMAGE_ARGS[@]}" \
    "${TEST_MASK_ARGS[@]}" \
    "${TEST_NAMES_ARGS[@]}" \
    --save_path "$SAVE_PATH" \
    --save_results "$SAVE_RESULTS" \
    --log_dir "$LOG_DIR" \
    --img_size 224 \
    --dino_pretrained "True" \
    --use_dilation "True" \
    --save_orig_size "False" \
    --device "$DEVICE"
