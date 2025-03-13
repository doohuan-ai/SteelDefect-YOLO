from ultralytics import YOLO

# 设置训练参数
def train_model():
    # 加载预训练模型
    model = YOLO("yolov8m.pt")
    
    # 设置训练参数
    results = model.train(
        data='data.yaml',                # 使用联合数据集配置文件
        epochs=250,                       # 训练轮数
        batch=16,                         # 训练批次大小
        project='/mnt/ssd/yolo/runs',     # 输出目录
        # 以下是可选参数，根据需要取消注释使用
        # imgsz=256,                        # 图像尺寸
        # patience=20,                      # 早停耐心值
        # device=0,                         # GPU设备
        # lr0=0.005,                        # 初始学习率
        # lrf=0.005,                        # 最终学习率
        # cls=2.0,                          # 分类损失权重
        # box=8.0,                          # 边界框损失权重
        # workers=4,                        # 数据加载工作线程数
        # val_batch=4,                      # 验证批次大小
        # cache=False,                      # 是否缓存图像
        # close_mosaic=0                    # 关闭马赛克增强的轮数
    )
    
    # 可选：将模型导出为ONNX格式
    # model.export(format='onnx')
    
    return results

if __name__ == '__main__':
    # 开始训练
    print("开始训练联合钢材表面缺陷检测模型...")
    results = train_model()
    print(f"训练完成！最佳mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"各类别性能:")
    for i, ap in enumerate(results.results_dict.get('metrics/mAP50-per-class', [])):
        print(f"    类别 {i}: {ap:.4f}") 