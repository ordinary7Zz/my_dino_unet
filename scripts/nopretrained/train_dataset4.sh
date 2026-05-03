#!/bin/bash

# 设置训练参数
CUDA_VISIBLE_DEVICES="1"
# 根据 CUDA_VISIBLE_DEVICES 自动设置 device，也可以手动指定（如 "cuda:0", "cpu"）
DEVICE="cuda:${CUDA_VISIBLE_DEVICES}"
METHOD="dino_unet"
TRAIN_IMAGE_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_experiment/dataset_4/images/"
TRAIN_MASK_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_experiment/dataset_4/masks/"
EPOCH=50
LR=0.0001
BATCH_SIZE=12
CHECKPOINT_DIR="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/nopretrained"
CHECKPOINT_INTERVAL=5
EVAL_INTERVAL=5
DATASET_NAME="nopretrained_train_dataset_4"

# 使用数组配置多个测试数据集
# 测试数据集名称数组
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

# 确保数组长度一致
if [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_IMAGE_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_MASK_PATHS[@]} ]; then
    echo "Error: Arrays must have the same length"
    exit 1
fi

# 构建命令参数
DATASET_NAMES_ARGS=""
IMAGE_PATHS_ARGS=""
MASK_PATHS_ARGS=""

for i in "${!TEST_DATASET_NAMES[@]}"; do
    DATASET_NAMES_ARGS="$DATASET_NAMES_ARGS ${TEST_DATASET_NAMES[$i]}"
done

for i in "${!TEST_IMAGE_PATHS[@]}"; do
    IMAGE_PATHS_ARGS="$IMAGE_PATHS_ARGS ${TEST_IMAGE_PATHS[$i]}"
done

for i in "${!TEST_MASK_PATHS[@]}"; do
    MASK_PATHS_ARGS="$MASK_PATHS_ARGS ${TEST_MASK_PATHS[$i]}"
done

# 执行训练脚本
python train.py \
    --method "$METHOD" \
    --train_image_path "$TRAIN_IMAGE_PATH" \
    --train_mask_path "$TRAIN_MASK_PATH" \
    --test_image_paths $IMAGE_PATHS_ARGS \
    --test_mask_paths $MASK_PATHS_ARGS \
    --test_dataset_names $DATASET_NAMES_ARGS \
    --epoch $EPOCH \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --dir_checkpoint "$CHECKPOINT_DIR" \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --dataset_name "$DATASET_NAME" \
    --img_size 224 \
    --dino_pretrained "False" \
    --device "$DEVICE"
