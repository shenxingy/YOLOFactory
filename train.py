#!/usr/bin/env python3
"""
Training script for fine-tuning a YOLOv8 model.

This script uses command-line arguments to configure the training process,
making it flexible and reusable for various experiments. It also includes
performance optimizations and an option to train on a fraction of the dataset for debugging.
"""
import argparse
import logging
from ultralytics import YOLO

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")

    # --- Model and Data Configuration ---
    parser.add_argument('--model-name', type=str, default='yolov8s.pt',
                        help='Pre-trained YOLOv8 model to start from (e.g., yolov8n.pt, yolov8s.pt).')
    parser.add_argument('--data-config', type=str, default='datasets/TESIS/data.yaml',
                        help='Path to the dataset configuration YAML file.')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=50,
                        help='Total number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=-1,
                        help='Batch size for training. -1 for auto-batch sizing.')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size for training.')
    parser.add_argument('--patience', type=int, default=0,
                        help='Number of epochs to wait before early stopping.')

    # --- Performance and Debugging ---
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads for data loading.')
    parser.add_argument('--cache-data', action='store_true',
                        help='Cache dataset images in RAM to accelerate training.')
    # NEW: Add fraction argument for debugging
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Fraction of the dataset to use for training (e.g., 0.1 for 10%%). Default is 1.0 (full dataset).')

    # --- Output and Logging ---
    parser.add_argument('--project-name', type=str, default='runs/train',
                        help='Directory to save training runs.')
    parser.add_argument('--run-name', type=str, default='TESIS_yolov8s',
                        help='Specific name for this training run.')
    
    # --- Hardware Configuration ---
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on, e.g., "0", "0,1", "cpu".')

    return parser.parse_args()


def main():
    """Main function to run the training process."""
    args = parse_args()
    logging.info("Starting YOLOv8 training with the following configuration:")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Dataset: {args.data_config}")
    logging.info(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, Image Size: {args.img_size}")
    logging.info(f"Performance: Workers={args.workers}, Caching={'Enabled' if args.cache_data else 'Disabled'}")
    # NEW: Log the fraction being used
    if args.fraction < 1.0:
        logging.warning(f"DEBUG MODE: Using only {args.fraction*100:.1f}% of the dataset.")

    try:
        model = YOLO(args.model_name)

        logging.info("Initiating model training...")
        model.train(
            data=args.data_config,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch_size,
            workers=args.workers,
            cache=args.cache_data,
            project=args.project_name,
            name=args.run_name,
            device=args.device,
            # NEW: Pass the fraction to the train function
            fraction=args.fraction,
            patience=args.patience
        )
        logging.info(f"Training complete! Model and results saved in '{args.project_name}/{args.run_name}'.")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)


if __name__ == '__main__':
    main()