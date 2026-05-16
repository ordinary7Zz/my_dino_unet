#!/bin/bash

# ---------------------- Configuration ----------------------
# Set CUDA device
CUDA_VISIBLE_DEVICES="0"
DEVICE="cuda:${CUDA_VISIBLE_DEVICES}"

# Model checkpoint path
CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/train_dataset_4/20260113_170943/dino_unet_train_dataset_4_epoch_50.pth"

# Test image root (recursive)
TEST_IMAGE_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/2016"

# Output root directory
OUTPUT_ROOT="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/screening_inference_outputs"

# Inference parameters
IMG_SIZE=224
BATCH_SIZE=4
NUM_WORKERS=4
DINO_PRETRAINED="True"
USE_DILATION="False"
SAVE_ORIG_SIZE="True"
THRESHOLD="0.5"
MIN_REGION_AREA=32
RANKING_SCORE="screen_score"
TOPK_SUMMARY=100
SAVE_BINARY_MASK="True"
SAVE_PROB_PNG="True"
SAVE_PROB_NPY="True"
SAVE_OVERLAY="True"
SAVE_INPUT_COPY="False"
EXPORT_PRIMARY_LESION_JSON="True"
PRIMARY_JSON_NAME="likely_primary_lesion_images.json"
PRIMARY_TOPK=0
PRIMARY_SORT_SCORE="screen_score"
EXPORT_REJECTED_JSON="False"
TRUST_THRESHOLDS_FILE="threshold_analysis_outputs/Malignant_ultrasound/thresholds_for_infer_recursive_screening.txt"
TRUST_FG_PROB_MEAN_MIN="0.60"
TRUST_PROB_MAX_MIN="0.80"
TRUST_LARGEST_COMPONENT_AREA_MIN=64
TRUST_POSITIVE_FRACTION_MIN="0.001"
TRUST_POSITIVE_FRACTION_MAX="0.35"
TRUST_LARGEST_COMPONENT_RATIO_MIN="0.50"
TRUST_NUM_COMPONENTS_MAX=5
TRUST_HIGH_CONF_FRACTION_0P9_MIN="0.0"

# ---------------------- Execution ----------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$REPO_ROOT/infer_recursive_screening.py"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file does not exist: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$TEST_IMAGE_PATH" ]; then
    echo "Error: Input directory does not exist: $TEST_IMAGE_PATH"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script does not exist: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

if [ -n "$TRUST_THRESHOLDS_FILE" ]; then
    case "$TRUST_THRESHOLDS_FILE" in
        /*) ;;
        *) TRUST_THRESHOLDS_FILE="$REPO_ROOT/$TRUST_THRESHOLDS_FILE" ;;
    esac
fi

mkdir -p "$OUTPUT_ROOT"

DATASET_NAME=$(basename "${TEST_IMAGE_PATH%/}")
OUTPUT_DIR="$OUTPUT_ROOT/${DATASET_NAME}_screening"
mkdir -p "$OUTPUT_DIR"

echo "Running screening inference for: $TEST_IMAGE_PATH"
echo "Saving outputs to: $OUTPUT_DIR"

CMD=(
    python -u "$PYTHON_SCRIPT"
    --checkpoint "$CHECKPOINT_PATH"
    --input_dir "$TEST_IMAGE_PATH"
    --output_dir "$OUTPUT_DIR"
    --img_size "$IMG_SIZE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --device "$DEVICE"
    --dino_pretrained "$DINO_PRETRAINED"
    --use_dilation "$USE_DILATION"
    --save_orig_size "$SAVE_ORIG_SIZE"
    --threshold "$THRESHOLD"
    --save_binary_mask "$SAVE_BINARY_MASK"
    --save_prob_png "$SAVE_PROB_PNG"
    --save_prob_npy "$SAVE_PROB_NPY"
    --save_overlay "$SAVE_OVERLAY"
    --save_input_copy "$SAVE_INPUT_COPY"
    --min_region_area "$MIN_REGION_AREA"
    --ranking_score "$RANKING_SCORE"
    --topk_summary "$TOPK_SUMMARY"
    --export_primary_lesion_json "$EXPORT_PRIMARY_LESION_JSON"
    --primary_json_name "$PRIMARY_JSON_NAME"
    --primary_topk "$PRIMARY_TOPK"
    --primary_sort_score "$PRIMARY_SORT_SCORE"
    --export_rejected_json "$EXPORT_REJECTED_JSON"
)

if [ -n "$TRUST_THRESHOLDS_FILE" ]; then
    if [ ! -f "$TRUST_THRESHOLDS_FILE" ]; then
        echo "Error: Trust thresholds file does not exist: $TRUST_THRESHOLDS_FILE"
        exit 1
    fi

    echo "Loading trust thresholds from: $TRUST_THRESHOLDS_FILE"
    mapfile -t TRUST_ARGS < <(
        python - "$TRUST_THRESHOLDS_FILE" <<'PY'
import pathlib
import shlex
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
text = text.replace("\\\r\n", " ").replace("\\\n", " ")
for token in shlex.split(text):
    print(token)
PY
    )
    CMD+=("${TRUST_ARGS[@]}")
else
    echo "Using inline trust thresholds from script configuration."
    CMD+=(
        --trust_fg_prob_mean_min "$TRUST_FG_PROB_MEAN_MIN"
        --trust_prob_max_min "$TRUST_PROB_MAX_MIN"
        --trust_largest_component_area_min "$TRUST_LARGEST_COMPONENT_AREA_MIN"
        --trust_positive_fraction_min "$TRUST_POSITIVE_FRACTION_MIN"
        --trust_positive_fraction_max "$TRUST_POSITIVE_FRACTION_MAX"
        --trust_largest_component_ratio_min "$TRUST_LARGEST_COMPONENT_RATIO_MIN"
        --trust_num_components_max "$TRUST_NUM_COMPONENTS_MAX"
        --trust_high_conf_fraction_0p9_min "$TRUST_HIGH_CONF_FRACTION_0P9_MIN"
    )
fi

"${CMD[@]}"
