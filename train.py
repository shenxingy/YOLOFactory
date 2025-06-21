from ultralytics import YOLO

def main():
    # 1. 加载一个预训练模型
    # 选择 'yolov8n.pt' (nano), 'yolov8s.pt' (small) 等
    # YOLOv8 会自动下载模型
    model = YOLO('yolov8s.pt')

    # 2. 指定数据集配置文件并开始训练
    # data: 我们准备好的 data.yaml 文件的路径
    # epochs: 训练轮次。对于迁移学习，从一个较小的数字开始，比如 50
    # imgsz: 输入图像的尺寸。640 是一个常用的值
    # batch: 批大小。根据你的显存（VRAM）调整。如果显存不足，减小这个值。-1 表示自动调整。
    # project: 训练结果保存的项目文件夹名
    # name: 本次训练的实验名
    results = model.train(
        data='datasets/TESIS/data.yaml',
        epochs=50,
        imgsz=640,
        batch=-1,
        project='runs/train',
        name='TESIS_yolov8s'
    )

    print("训练完成！模型和结果保存在 'runs/train/TESIS_yolov8s' 文件夹中。")

if __name__ == '__main__':
    main()