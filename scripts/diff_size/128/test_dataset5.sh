#!/bin/bash

# ---------------------- Configuration ----------------------
# Set CUDA device
CUDA_VISIBLE_DEVICES="1"
# 根据 CUDA_VISIBLE_DEVICES 自动设置 device，也可以手动指定（如 "cuda:0", "cpu"）
DEVICE="cuda:${CUDA_VISIBLE_DEVICES}"

IMG_SIZE=128

# Model checkpoint path
CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/diff_size/448/diff_size_train_dataset_5/20251224_230237/dino_unet_diff_size_train_dataset_5_epoch_50.pth"

# Configure multiple test dataset paths
# Test dataset names array
TEST_DATASET_NAMES=(
    "TN3K"
    "DDTI"
    "ThyroidXL"
    "PKTN"
    "TN5K"
)

# 测试图像路径数组
TEST_IMAGE_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TN3K/test-image/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/DDTI/2_preprocessed_data/stage1/p_image/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/ThyroidXL/test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/PKTN_processed/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TN5K_processed/test/images/"
)

# 测试掩码路径数组
TEST_MASK_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TN3K/test-mask/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/DDTI/2_preprocessed_data/stage1/p_mask/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/ThyroidXL/test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/PKTN_processed/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TN5K_processed/test/masks/"
)

# Ensure arrays have the same length
if [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_IMAGE_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_MASK_PATHS[@]} ]; then
    echo "Error: Arrays must have the same length"
    exit 1
fi

# Prediction results save path
SAVE_PATH="./predictions/test_dataset5"

# Whether to save prediction results (true/false)
SAVE_RESULTS="false"

# Log directory
LOG_DIR="./logs/test_logs/diff_size/$IMG_SIZE/test_dataset5"

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

# Choose test script based on whether to use Dilation
USE_Dilation="false"

if [ "$USE_Dilation" = "true" ] ; then
    TEST_SCRIPT="test_parallel_Dilation.py"
else
    TEST_SCRIPT="test_parallel.py"
fi
# Execute the test command
python -u "$TEST_SCRIPT" \
    --checkpoint "$CHECKPOINT_PATH" \
    "${TEST_IMAGE_ARGS[@]}" \
    "${TEST_MASK_ARGS[@]}" \
    "${TEST_NAMES_ARGS[@]}" \
    --save_path "$SAVE_PATH" \
    --save_results "$SAVE_RESULTS" \
    --log_dir "$LOG_DIR" \
    --img_size $IMG_SIZE \
    --device "$DEVICE"
