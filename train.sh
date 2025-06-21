python train.py \
    --model-name yolov8m.pt \
    --epochs 50 \
    --batch-size 32 \
    --workers 16 \
    --run-name TESIS_yolov8m_50_epochs \
    --cache-data \
    > train.log 2>&1 &
    