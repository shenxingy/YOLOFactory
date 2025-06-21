#!/bin/bash

# ==============================================================================
#  YOLOv8 Prediction Runner Script
#
#  This script provides an easy way to run inference using the predict.py script.
#  Simply configure the variables in the section below and execute the script.
# ==============================================================================

# --- Configuration ---
# Set this to the path of the weights file from your best training run.
WEIGHTS_PATH="runs/train/TESIS_yolov8s/weights/best.pt"

# Set this to the image, video, or folder you want to run prediction on.
# IMPORTANT: CHANGE THIS to a real image or video file for testing.
INPUT_SOURCE="path/to/your/test_image.jpg" 

# Set the directory where you want to save the output.
OUTPUT_DIR="runs/predict"

# Set the confidence threshold for detections.
CONFIDENCE_THRESHOLD=0.4
# --- End of Configuration ---


# Check if the input source exists
if [ ! -e "$INPUT_SOURCE" ]; then
    echo "Error: Input source '$INPUT_SOURCE' not found."
    echo "Please update the INPUT_SOURCE variable in this script."
    exit 1
fi

echo "--- Running YOLOv8 Prediction ---"
echo "  Weights:    ${WEIGHTS_PATH}"
echo "  Source:     ${INPUT_SOURCE}"
echo "  Confidence: ${CONFIDENCE_THRESHOLD}"
echo "-----------------------------------"

# Activate your virtual environment if you have one
# source venv/bin/activate

python predict.py \
    --weights "${WEIGHTS_PATH}" \
    --source "${INPUT_SOURCE}" \
    --output-dir "${OUTPUT_DIR}" \
    --conf-thres ${CONFIDENCE_THRESHOLD} \
    --save

echo "--- Prediction Finished ---"
echo "Results have been saved to the '${OUTPUT_DIR}' directory."