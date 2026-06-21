#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/dino_unet_train_dataset_4_epoch_50.pth"
OUTPUT_DIR="./pyradiomics_dice/gradcam/BM"
IMG_SIZE="224"
DINO_PRETRAINED="True"
ALPHA="0.3"
TARGET_LAYER="up1"
SMOOTH_SIGMA_RATIO="0.1"
GAMMA="0.4"
SATURATION_SCALE="1.5"
DEFAULT_OUTPUT_TYPE="all"

COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --output_dir "$OUTPUT_DIR"
    --img_size "$IMG_SIZE"
    --dino_pretrained "$DINO_PRETRAINED"
    --alpha "$ALPHA"
    --target_layer "$TARGET_LAYER"
    --smooth_sigma_ratio "$SMOOTH_SIGMA_RATIO"
    --gamma "$GAMMA"
    --saturation_scale "$SATURATION_SCALE"
)

run_case() {
    local category="$1"
    local image_path="$2"
    local mask_path="$3"
    local output_type="${4:-$DEFAULT_OUTPUT_TYPE}"

    echo "Running case: ${category}"
    echo "  image: ${image_path}"
    echo "  mask : ${mask_path}"

    python gradcam_single_image_seg.py \
        "${COMMON_ARGS[@]}" \
        --image_path "$image_path" \
        --mask_path "$mask_path" \
        --output_type "$output_type"
}

while IFS='|' read -r category image_path mask_path output_type; do
    if [[ -z "${category// }" ]]; then
        continue
    fi

    if [[ "$category" == \#* ]]; then
        echo "$category"
        continue
    fi

    run_case "$category" "$image_path" "$mask_path" "$output_type"
done <<'EOF'
# Benign good masks
Benign good masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/train/images/ThyroidXL_train_00002730_C9690598_1.png|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/train/masks/ThyroidXL_train_00002730_C9690598_1.png|all

# Benign bad masks
Benign bad masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/TN3K_test_0040.jpg|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks/TN3K_test_0040.jpg|all

# Malignant good masks
Malignant good masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images/TN5K_test_003323.jpg|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/masks/TN5K_test_003323.png|all

# Malignant bad masks
Malignant bad masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/ThyroidXL_test_00001873_6923593C_2.png|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/ThyroidXL_test_00001873_6923593C_2.png|all
Malignant bad masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/ThyroidXL_test_00001978_DC398883_0.png|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/ThyroidXL_test_00001978_DC398883_0.png|all
Malignant bad masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/ThyroidXL_test_00003932_89B4CFAD_1.png|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/ThyroidXL_test_00003932_89B4CFAD_1.png|all
Malignant bad masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/ThyroidXL_test_00001378_BA5E9CC4_0.png|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/ThyroidXL_test_00001378_BA5E9CC4_0.png|all
Malignant bad masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/TN3K_test_0586.jpg|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks/TN3K_test_0586.jpg|all
Malignant bad masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/ThyroidXL_test_00002838_EDBD208B_2.png|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/ThyroidXL_test_00002838_EDBD208B_2.png|all
Malignant bad masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/ThyroidXL_test_00002838_1E638EAB_1.png|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/ThyroidXL_test_00002838_1E638EAB_1.png|
Malignant bad masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/ThyroidXL_test_00002755_F79615B3_0.png|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/ThyroidXL_test_00002755_F79615B3_0.png|all
Malignant bad masks|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/ThyroidXL_test_00002838_A9C56A4B_0.png|/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/ThyroidXL_test_00002838_A9C56A4B_0.png|all
EOF
