#!/usr/bin/env python3
"""
Professional prediction script for running inference with a trained YOLOv8 model.

This script takes a trained model's weights and an input source (image, video, or folder)
and runs prediction, saving the annotated results to a specified directory.
"""
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    """Parses command-line arguments for the prediction script."""
    parser = argparse.ArgumentParser(description="YOLOv8 Prediction Script")

    # --- Core Arguments ---
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the trained model weights (.pt file).')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the input source (image, video, or directory).')
    
    # --- Output Configuration ---
    parser.add_argument('--output-dir', type=str, default='runs/predict',
                        help='Directory to save prediction results.')
    parser.add_argument('--save', action='store_true',
                        help='Explicitly save the output images/videos with annotations.')

    # --- Inference Parameters ---
    parser.add_argument('--conf-thres', type=float, default=0.4,
                        help='Confidence threshold for detections.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on, e.g., "0", "0,1", "cpu".')
    
    # --- Visualization Options ---
    parser.add_argument('--hide-labels', action='store_true',
                        help='Hide class labels on the output.')
    parser.add_argument('--hide-conf', action='store_true',
                        help='Hide confidence scores on the output.')

    return parser.parse_args()


def main():
    """Main function to run the prediction process."""
    args = parse_args()
    logging.info("Starting YOLOv8 prediction with the following configuration:")
    logging.info(f"Weights: {args.weights}")
    logging.info(f"Source: {args.source}")
    logging.info(f"Output Directory: {args.output_dir}")
    logging.info(f"Confidence Threshold: {args.conf_thres}")

    # --- Validate paths ---
    weights_path = Path(args.weights)
    source_path = Path(args.source)
    if not weights_path.exists():
        logging.error(f"Error: Weights file not found at '{weights_path}'")
        return
    if not source_path.exists():
        logging.error(f"Error: Input source not found at '{source_path}'")
        return

    try:
        # Load the trained YOLOv8 model
        model = YOLO(weights_path)

        # Run prediction
        logging.info("Running inference...")
        results = model.predict(
            source=str(source_path),
            conf=args.conf_thres,
            save=args.save,
            project=args.output_dir,
            name='latest_run', # This will create a directory like 'runs/predict/latest_run'
            exist_ok=True, # Allows overwriting previous 'latest_run'
            device=args.device,
            hide_labels=args.hide_labels,
            hide_conf=args.hide_conf
        )
        
        # The results object contains detailed information, but the visual output is saved automatically if save=True
        num_detections = sum(len(r.boxes) for r in results)
        logging.info(f"Prediction complete. Found a total of {num_detections} objects.")
        if args.save:
            logging.info(f"Results saved to '{args.output_dir}/latest_run'")
        else:
            logging.info("To save visual results, please run with the --save flag.")

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)


if __name__ == '__main__':
    main()