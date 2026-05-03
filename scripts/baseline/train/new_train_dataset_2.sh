#!/bin/bash

# 设置训练参数
Train_DATASET="dataset_2"
CUDA_VISIBLE_DEVICES="0"
# 根据 CUDA_VISIBLE_DEVICES 自动设置 device，也可以手动指定（如 "cuda:0", "cpu"）
DEVICE="cuda:${CUDA_VISIBLE_DEVICES}"
METHOD="dino_unet"
TRAIN_IMAGE_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/${Train_DATASET}/train/images/"
TRAIN_MASK_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/${Train_DATASET}/train/masks/"
EPOCH=50
LR=0.0001
BATCH_SIZE=12
CHECKPOINT_DIR="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline"
CHECKPOINT_INTERVAL=10
EVAL_INTERVAL=10
DATASET_NAME="train_${Train_DATASET}"

# 使用数组配置多个测试数据集
# 测试数据集名称数组
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
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/valid/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI/valid/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/valid/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/PKTN/valid/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/valid/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/valid/images/"
)

# 测试掩码路径数组
TEST_MASK_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/valid/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI/valid/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/valid/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/PKTN/valid/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/valid/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/valid/masks/"
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
    --device "$DEVICE"