import os
import json
import glob
from pathlib import Path
import shutil
import random

# 设置随机种子以确保可重复性
random.seed(42)

# 配置
src_dir = "/mnt/hdd/datasets/steel-surface-defect-sample/GC10-DET"  # 原始数据集位置
src_img_dir = os.path.join(src_dir, "ds/img")  # 原始图像位置
src_ann_dir = os.path.join(src_dir, "ds/ann")  # 原始标注位置
dst_dir = "datasets/GC10-DET"  # 目标目录
train_ratio = 0.8  # 80%用于训练，20%用于验证

# 清空目标目录（如果存在）
def clean_dst_dir():
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

# 类别映射
meta_file = os.path.join(src_dir, "meta.json")
with open(meta_file, 'r') as f:
    meta_data = json.load(f)

# 创建类别ID映射
class_names = [cls["title"] for cls in meta_data["classes"]]
class_mapping = {name: i+6 for i, name in enumerate(class_names)}  # 从6开始，因为NEU-DET已经有6个类别

# 创建必要的目录
def create_dirs():
    dirs = [
        f"{dst_dir}/images/train",
        f"{dst_dir}/images/val",
        f"{dst_dir}/labels/train",
        f"{dst_dir}/labels/val"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# 转换标注文件为YOLO格式
def convert_annotations():
    # 获取所有json文件
    ann_files = glob.glob(f"{src_ann_dir}/*.json")
    
    # 分割训练集和验证集
    random.shuffle(ann_files)
    split_idx = int(len(ann_files) * train_ratio)
    train_files = ann_files[:split_idx]
    val_files = ann_files[split_idx:]
    
    # 处理训练集
    process_files(train_files, "train")
    
    # 处理验证集
    process_files(val_files, "val")
    
    print(f"转换完成，处理了{len(ann_files)}个文件")
    print(f"训练集: {len(train_files)}个文件")
    print(f"验证集: {len(val_files)}个文件")

# 处理文件
def process_files(file_list, subset):
    successful = 0
    skipped = 0
    
    for ann_file in file_list:
        # 读取标注文件
        with open(ann_file, 'r') as f:
            ann_data = json.load(f)
        
        # 获取图像文件名（不重复添加.jpg扩展名）
        base_name = os.path.basename(ann_file).replace('.json', '')
        img_name = base_name
        if not img_name.endswith('.jpg'):
            img_name = f"{img_name}.jpg"
        
        # 获取图像尺寸
        img_width = ann_data.get("size", {}).get("width", 0)
        img_height = ann_data.get("size", {}).get("height", 0)
        
        if img_width == 0 or img_height == 0:
            print(f"警告: 图像 {img_name} 尺寸未知，跳过")
            skipped += 1
            continue
        
        # 复制图像
        src_img = os.path.join(src_img_dir, img_name)
        dst_img = os.path.join(dst_dir, "images", subset, img_name)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f"警告: 图像 {img_name} 不存在，跳过")
            skipped += 1
            continue
        
        # 创建YOLO格式的标签文件
        base_name = os.path.splitext(img_name)[0]  # 移除扩展名
        dst_label = os.path.join(dst_dir, "labels", subset, f"{base_name}.txt")
        
        with open(dst_label, 'w') as f:
            # 处理每个对象
            for obj in ann_data.get("objects", []):
                class_name = obj.get("classTitle")
                if class_name not in class_mapping:
                    print(f"警告: 未知类别 {class_name}, 跳过")
                    continue
                
                class_id = class_mapping[class_name]
                bbox = obj.get("points", {}).get("exterior", [])
                
                if len(bbox) != 2:
                    print(f"警告: 对象 {obj} 边界框格式不正确，跳过")
                    continue
                
                # 计算YOLO格式坐标（中心点x, 中心点y, 宽度, 高度，都是相对值）
                x_min, y_min = bbox[0]
                x_max, y_max = bbox[1]
                
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # 写入YOLO格式：类别 中心x 中心y 宽度 高度
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        successful += 1
    
    print(f"{subset}集处理: 成功={successful}, 跳过={skipped}")

if __name__ == "__main__":
    print("开始转换GC10-DET数据集...")
    clean_dst_dir()
    create_dirs()
    convert_annotations()
    print("转换完成！") 