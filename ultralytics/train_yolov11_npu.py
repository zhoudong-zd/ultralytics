# 创建 train_yolov11_npu.py
cat > train_yolov11_npu.py <<EOF
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 NPU Training')
    parser.add_argument('--weights', type=str, default='yolov11s.pt', help='初始权重')
    parser.add_argument('--data', type=str, default='coco.yaml', help='数据集配置')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--device', type=str, default='npu:0', help='设备')
    parser.add_argument('--sample-ratio', type=float, default=0.05, help='数据采样比例')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    
    args = parser.parse_args()
    
    # 加载模型
    model = YOLO(args.weights)
    
    # 训练配置
    train_kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'device': args.device,
        'batch': args.batch_size,
        'amp': True,  # 混合精度
        'project': 'runs/train',
        'name': 'exp',
        'exist_ok': True,
        'sample_ratio': args.sample_ratio  # 自定义采样参数
    }
    
    # 启动训练
    results = model.train(**train_kwargs)
    print(f"训练完成! 最佳模型保存在: {results.save_dir}")

if __name__ == '__main__':
    main()
EOF
