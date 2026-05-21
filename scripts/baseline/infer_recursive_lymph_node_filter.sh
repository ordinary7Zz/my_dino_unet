#!/bin/bash

# ---------------------- Configuration ----------------------
CUDA_VISIBLE_DEVICES="0"
DEVICE="cuda:${CUDA_VISIBLE_DEVICES}"

CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/train_Lymph_Node_Metastasis/Lymph_Node_Metastasis/20260521_211141/dino_unet_Lymph_Node_Metastasis_epoch_40.pth"
INPUT_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/"
OUTPUT_JSON="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_lymph_node_filtered.json"
DEBUG_OUTPUT_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_lymph_node_filtered_debug"

IMG_SIZE=224
BATCH_SIZE=4
NUM_WORKERS=4
DINO_PRETRAINED="True"
USE_DILATION="False"
SAVE_ORIG_SIZE="True"
THRESHOLD="0.5"
MIN_REGION_AREA=32
MIN_COMPONENT_AREA=64
PREFERRED_COMPONENT_FRACTION_MIN="0.002"
PREFERRED_COMPONENT_FRACTION_PEAK_HIGH="0.12"
MAX_COMPONENT_FRACTION="0.25"
MAX_COMPONENTS_REJECT=5
MIN_LARGEST_COMPONENT_RATIO="0.50"
MIN_COMPONENT_FILL_RATIO="0.20"
MAX_CENTER_DISTANCE="0.65"
AGREEMENT_RING_WIDTH=12
VALID_ULTRASOUND_BLACK_THRESHOLD=8
MIN_VALID_ULTRASOUND_FRACTION="0.80"
BORDER_TOUCH_LARGE_COMPONENT_FRACTION="0.03"
SELECTION_SCORE_THRESHOLD="0.60"
MIN_AGREEMENT_SCORE="0.30"
SAVE_BINARY_MASK="False"
SAVE_PROB_PNG="False"
SAVE_PROB_NPY="False"
SAVE_OVERLAY="False"
INCLUDE_REJECTED_DETAILS="False"

# ---------------------- Execution ----------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$REPO_ROOT/infer_recursive_lymph_node_filter.py"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file does not exist: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script does not exist: $PYTHON_SCRIPT"
    exit 1
fi

mkdir -p "$DEBUG_OUTPUT_DIR"

if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

CMD=(
    python -u "$PYTHON_SCRIPT"
    --checkpoint "$CHECKPOINT_PATH"
    --input_dir "$INPUT_DIR"
    --output_json "$OUTPUT_JSON"
    --img_size "$IMG_SIZE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --device "$DEVICE"
    --dino_pretrained "$DINO_PRETRAINED"
    --use_dilation "$USE_DILATION"
    --save_orig_size "$SAVE_ORIG_SIZE"
    --threshold "$THRESHOLD"
    --min_region_area "$MIN_REGION_AREA"
    --min_component_area "$MIN_COMPONENT_AREA"
    --preferred_component_fraction_min "$PREFERRED_COMPONENT_FRACTION_MIN"
    --preferred_component_fraction_peak_high "$PREFERRED_COMPONENT_FRACTION_PEAK_HIGH"
    --max_component_fraction "$MAX_COMPONENT_FRACTION"
    --max_components_reject "$MAX_COMPONENTS_REJECT"
    --min_largest_component_ratio "$MIN_LARGEST_COMPONENT_RATIO"
    --min_component_fill_ratio "$MIN_COMPONENT_FILL_RATIO"
    --max_center_distance "$MAX_CENTER_DISTANCE"
    --agreement_ring_width "$AGREEMENT_RING_WIDTH"
    --valid_ultrasound_black_threshold "$VALID_ULTRASOUND_BLACK_THRESHOLD"
    --min_valid_ultrasound_fraction "$MIN_VALID_ULTRASOUND_FRACTION"
    --border_touch_large_component_fraction "$BORDER_TOUCH_LARGE_COMPONENT_FRACTION"
    --selection_score_threshold "$SELECTION_SCORE_THRESHOLD"
    --min_agreement_score "$MIN_AGREEMENT_SCORE"
    --debug_output_dir "$DEBUG_OUTPUT_DIR"
    --save_binary_mask "$SAVE_BINARY_MASK"
    --save_prob_png "$SAVE_PROB_PNG"
    --save_prob_npy "$SAVE_PROB_NPY"
    --save_overlay "$SAVE_OVERLAY"
    --include_rejected_details "$INCLUDE_REJECTED_DETAILS"
)

"${CMD[@]}"
