#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
此脚本用于使用训练好的YOLOv8模型检测钢材表面的缺陷。
支持对单张图片、多张图片、视频文件和摄像头流进行实时检测。
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 添加颜色常量
COLORS = {
    'success': '\033[92m',  # 绿色
    'info': '\033[94m',     # 蓝色
    'warning': '\033[93m',  # 黄色
    'error': '\033[91m',    # 红色
    'end': '\033[0m'        # 重置颜色
}

def print_color(text, color='info'):
    """带颜色地打印文本"""
    print(f"{COLORS.get(color, COLORS['info'])}{text}{COLORS['end']}")

class SteelDefectDetector:
    """钢材表面缺陷检测器类"""
    
    def __init__(self, weights, conf_thres=0.25, iou_thres=0.45, device=''):
        """
        初始化钢材表面缺陷检测器
        
        参数:
            weights (str): 模型权重文件路径
            conf_thres (float): 置信度阈值
            iou_thres (float): IoU阈值
            device (str): 使用的设备 ('cuda', 'cpu' 或 '')
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 检查权重文件是否存在
        if not os.path.exists(weights):
            print_color(f"错误: 权重文件 '{weights}' 不存在!", 'error')
            sys.exit(1)
        
        print_color(f"加载模型: {weights}", 'info')
        try:
            # 加载YOLOv8模型
            self.model = YOLO(weights)
            self.device = device
            self.names = self.model.names  # 类别名称
            print_color(f"模型加载成功! 可检测的缺陷类型: {list(self.names.values())}", 'success')
        except Exception as e:
            print_color(f"模型加载失败: {e}", 'error')
            sys.exit(1)
    
    def detect(self, source, save_dir='results', view_img=True, save_txt=False, save_conf=False):
        """
        在指定源上运行检测
        
        参数:
            source (str): 输入源 (图片文件、目录、视频文件或摄像头序号)
            save_dir (str): 结果保存目录
            view_img (bool): 是否显示结果
            save_txt (bool): 是否保存检测结果为txt文件
            save_conf (bool): 是否在txt结果中包含置信度
        """
        # 创建保存目录
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        source = str(source)
        is_image = source.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp'))
        is_video = source.lower().endswith(('mp4', 'mov', 'avi', 'mkv', 'flv', 'webm'))
        is_camera = source.isdigit()
        
        # 如果是视频或摄像头
        if is_video or is_camera:
            self._process_video(source, save_dir, view_img, save_txt, save_conf)
        # 如果是图像
        elif is_image or os.path.isdir(source):
            self._process_image(source, save_dir, view_img, save_txt, save_conf)
        else:
            print_color(f"不支持的输入源: {source}", 'error')
            sys.exit(1)
    
    def _process_image(self, source, save_dir, view_img, save_txt, save_conf):
        """处理图像"""
        print_color(f"处理图像: {source}", 'info')
        
        # 运行推理
        start_time = time.time()
        results = self.model.predict(
            source=source,
            conf=self.conf_thres,
            iou=self.iou_thres,
            save=True,
            save_txt=save_txt,
            save_conf=save_conf,
            project=str(save_dir.parent),
            name=save_dir.name,
            exist_ok=True
        )
        end_time = time.time()
        
        # 处理结果
        for i, result in enumerate(results):
            img_path = result.path
            img_name = Path(img_path).name
            detections = result.boxes
            
            # 统计类别信息
            class_counts = {}
            for det in detections:
                cls = int(det.cls[0].item())
                class_name = self.names[cls]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # 打印结果
            if len(detections) > 0:
                print_color(f"图像 {img_name} 检测到 {len(detections)} 个缺陷:", 'success')
                for cls_name, count in class_counts.items():
                    print_color(f"  - {cls_name}: {count}个", 'info')
            else:
                print_color(f"图像 {img_name} 没有检测到缺陷", 'warning')
        
        print_color(f"处理完成! 耗时: {end_time - start_time:.2f}秒", 'success')
    
    def _process_video(self, source, save_dir, view_img, save_txt, save_conf):
        """处理视频或摄像头"""
        is_camera = source.isdigit()
        if is_camera:
            source = int(source)
            print_color(f"打开摄像头 ID: {source}", 'info')
        else:
            print_color(f"处理视频: {source}", 'info')
        
        # 运行推理
        try:
            self.model.predict(
                source=source,
                conf=self.conf_thres,
                iou=self.iou_thres,
                show=view_img,
                save=True,
                save_txt=save_txt,
                save_conf=save_conf,
                project=str(save_dir.parent),
                name=save_dir.name,
                exist_ok=True,
                stream=True
            )
            print_color(f"视频处理完成!", 'success')
        except KeyboardInterrupt:
            print_color("用户中断处理", 'warning')
        except Exception as e:
            print_color(f"视频处理出错: {e}", 'error')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='钢材表面缺陷检测')
    parser.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt', help='模型权重文件路径')
    parser.add_argument('--source', type=str, default='0', help='输入源 (图片文件/目录/视频/摄像头ID)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='', help='计算设备 (cuda设备, 如 0 或 0,1,2,3 或 cpu)')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='保存结果到 *.txt')
    parser.add_argument('--save-conf', action='store_true', help='在txt结果中保存置信度')
    parser.add_argument('--project', type=str, default='runs/detect', help='结果保存到项目/名称')
    parser.add_argument('--name', type=str, default='exp', help='结果保存到项目/名称')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    print_color("钢材表面缺陷检测系统", 'info')
    print_color("-" * 50, 'info')
    
    # 显示参数
    print_color("参数设置:", 'info')
    for k, v in vars(args).items():
        print_color(f"  {k}: {v}", 'info')
    print_color("-" * 50, 'info')
    
    # 初始化检测器
    detector = SteelDefectDetector(
        weights=args.weights,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device
    )
    
    # 运行检测
    save_dir = Path(args.project) / args.name
    detector.detect(
        source=args.source,
        save_dir=save_dir,
        view_img=args.view_img,
        save_txt=args.save_txt,
        save_conf=args.save_conf
    )

if __name__ == '__main__':
    main() 