# Debug mode
# python train.py \
#     --fraction 0.05 \
#     --epochs 3 \
#     --run-name "debug_run_5_percent" \
#     --batch-size 8 \
#     --workers 4

# Train mode
# python train.py \
#     --model-name 'yolov8m.pt' \
#     --data-config 'datasets/TESIS/data.yaml' \
#     --epochs 50 \
#     --batch-size 16 \
#     --img-size 640 \
#     --workers 8 \
#     --cache-data \
#     --run-name 'TESIS_yolov8s_full_run_01' \
#     --patience 10 \
#     > training_full.log 2>&1 &
    
python train.py \
    --model-name 'runs/train/TESIS_yolov8s_full_run_01/weights/last.pt' \
    --data-config 'datasets/TESIS/data.yaml' \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --workers 8 \
    --cache-data \
    --run-name 'TESIS_yolov8s_full_run_01_resume' \
    --patience 10 \
    --device 0 \
    > training_resume.log 2>&1 &
   