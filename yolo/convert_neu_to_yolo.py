import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path
import glob
import argparse

# 设置随机种子以确保可重复性
random.seed(42)

# 类别映射（NEU-DET 有 6 类）
CLASS_MAPPING = {
    "rolled-in_scale": 0,
    "patches": 1,
    "crazing": 2,
    "pitted_surface": 3,
    "inclusion": 4,
    "scratches": 5
}

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='将NEU-DET数据集转换为YOLO格式')
    parser.add_argument('src_dir', type=str, 
                        help='原始NEU-DET数据集位置（必传参数）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例，默认0.8表示80%用于训练，20%用于验证')
    return parser.parse_args()

# 配置参数
args = parse_args()
src_dir = args.src_dir  # 原始数据集位置
src_xml_dir = os.path.join(src_dir, "annotations")  # 原始XML标注位置
src_img_dir = os.path.join(src_dir, "images")  # 原始图像位置
dst_dir = "datasets/NEU-DET"  # 转换后的数据集保存位置
train_ratio = args.train_ratio  # 80%用于训练，20%用于验证

# 清空目标目录（如果存在）
def clean_dst_dir():
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

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

# 查找图像文件（支持多种扩展名）
def find_image_file(base_name):
    # 支持的图像扩展名
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for ext in extensions:
        # 尝试完整路径
        full_path = os.path.join(src_img_dir, base_name + ext)
        if os.path.exists(full_path):
            return full_path
            
    # 如果基本名称中已经包含扩展名
    if os.path.exists(os.path.join(src_img_dir, base_name)):
        return os.path.join(src_img_dir, base_name)
            
    # 尝试查找匹配的文件
    pattern = os.path.join(src_img_dir, f"{base_name}.*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
        
    return None

# 转换XML为YOLO格式并生成标签文件
def convert_xml_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取基本文件名（可能有或没有扩展名）
    img_filename = root.find("filename").text
    
    # 移除可能的扩展名，以获取纯基本名称
    base_name = os.path.splitext(img_filename)[0]
    
    image_w = int(root.find("size/width").text)
    image_h = int(root.find("size/height").text)

    yolo_boxes = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text.lower()  # 转换为小写
        if class_name not in CLASS_MAPPING:
            print(f"警告：未知的类别名称 '{class_name}' 在文件 {xml_file} 中")
            continue

        class_id = CLASS_MAPPING[class_name]
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # 计算 YOLO 格式的坐标
        x_center = (xmin + xmax) / 2 / image_w
        y_center = (ymin + ymax) / 2 / image_h
        width = (xmax - xmin) / image_w
        height = (ymax - ymin) / image_h

        yolo_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return base_name, yolo_boxes

# 获取所有XML文件
def get_all_xml_files():
    xml_files = []
    for xml_file in os.listdir(src_xml_dir):
        if xml_file.endswith(".xml"):
            xml_files.append(os.path.join(src_xml_dir, xml_file))
    return xml_files

# 处理并分割数据集
def process_and_split_dataset():
    # 获取所有XML文件
    xml_files = get_all_xml_files()
    
    # 随机打乱
    random.shuffle(xml_files)
    
    # 分割训练集和验证集
    split_idx = int(len(xml_files) * train_ratio)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]
    
    # 处理训练集
    process_files(train_files, "train")
    
    # 处理验证集
    process_files(val_files, "val")
    
    print(f"处理完成，共处理了{len(xml_files)}个文件")
    print(f"训练集: {len(train_files)}个样本")
    print(f"验证集: {len(val_files)}个样本")

# 处理文件集合
def process_files(file_list, subset):
    successful = 0
    skipped = 0
    
    for xml_file in file_list:
        base_name, yolo_boxes = convert_xml_to_yolo(xml_file)
        
        # 跳过没有标注的样本
        if not yolo_boxes:
            skipped += 1
            continue
            
        # 查找并复制图像
        src_img = find_image_file(base_name)
        if not src_img:
            print(f"警告: 找不到图像 {base_name}，跳过")
            skipped += 1
            continue
            
        # 获取图像文件名和扩展名
        img_filename = os.path.basename(src_img)
        dst_img = os.path.join(dst_dir, "images", subset, img_filename)
        
        # 复制图像
        shutil.copy(src_img, dst_img)
        
        # 保存YOLO格式标签
        dst_label = os.path.join(dst_dir, "labels", subset, f"{base_name}.txt")
        
        with open(dst_label, "w") as f:
            for box in yolo_boxes:
                f.write(f"{box}\n")
                
        successful += 1
    
    print(f"{subset}集处理: 成功={successful}, 跳过={skipped}")

if __name__ == "__main__":
    print("开始处理NEU-DET数据集...")
    clean_dst_dir()
    create_dirs()
    process_and_split_dataset()
    print("NEU-DET数据集处理完成！")
