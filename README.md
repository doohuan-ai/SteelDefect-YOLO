# 钢材表面缺陷检测系统

<div align="center">
  <img src="images/doohuan_logo.png" alt="多焕智能Logo" width="300"/>
  <h1>多焕智能（DooHuan AI）</h1>
  <p>
    <b>工业智能视觉检测系统</b>
  </p>
  <p>
    <a href="https://www.doohuan.com">
      <img alt="官网" src="https://img.shields.io/badge/官网-doohuan.com-blue?style=flat-square" />
    </a>
    <a href="https://github.com/doohuan-ai">
      <img alt="GitHub" src="https://img.shields.io/badge/GitHub-doohuan--ai-lightgrey?style=flat-square&logo=github" />
    </a>
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  </p>
  <hr>
</div>

基于YOLOv8的钢材表面缺陷检测系统，使用NEU-DET和GC10-DET两个公开数据集，能够检测16种不同类型的钢材表面缺陷。

## 项目特点

- 使用最新的YOLOv8目标检测模型
- 联合两个专业钢材表面缺陷数据集
- 高精度检测16种常见钢材表面缺陷类型
- 支持实时检测和批量分析

## 数据集介绍

本项目使用了两个公开的钢材表面缺陷数据集：

1. **NEU-DET**：东北大学钢材表面缺陷数据集
   - 6种常见缺陷类型：轧制鳞片、斑块、裂纹、点蚀表面、夹杂物和划痕
   - 每种缺陷300张图像，共1800张图像
   - 图像尺寸：200×200像素

2. **GC10-DET**：钢板表面缺陷数据集
   - 10种缺陷类型：冲孔、焊接线、新月形缝隙、水斑、油斑、丝状斑点、夹杂、轧制坑、折痕和腰褶
   - 共2300张高质量标注图像

## 文件夹结构

```
├── datasets/                   # 数据集目录
│   ├── NEU-DET/               # NEU-DET数据集
│   │   ├── images/            # 图像文件
│   │   │   ├── train/         # 训练集图像
│   │   │   └── val/           # 验证集图像
│   │   └── labels/            # YOLO格式标签
│   │       ├── train/         # 训练集标签
│   │       └── val/           # 验证集标签
│   └── GC10-DET/              # GC10-DET数据集
│       ├── images/            # 图像文件
│       │   ├── train/         # 训练集图像
│       │   └── val/           # 验证集图像
│       └── labels/            # YOLO格式标签
│           ├── train/         # 训练集标签
│           └── val/           # 验证集标签
├── convert_neu_to_yolo.py     # NEU-DET数据集转换脚本
├── convert_gc10_to_yolo.py    # GC10-DET数据集转换脚本
├── data.yaml                  # 联合数据集配置文件
├── train.py                   # 训练脚本
├── detect.py                  # 目标检测推理脚本（待添加）
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明文档
```

## 安装与使用

### 环境配置

1. 克隆仓库：
```bash
git clone https://github.com/doohuan-ai/steel-surface-defect-detection.git
cd steel-surface-defect-detection
```

2. 创建并激活环境：
```bash
# 使用conda
conda create -n steel-surface-defect-detection python=3.9
conda activate steel-surface-defect-detection

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

### 数据集准备

如果您已有NEU-DET和GC10-DET数据集：

1. 将原始数据集转换为YOLO格式：
```bash
python convert_neu_to_yolo.py
python convert_gc10_to_yolo.py
```

如果您没有数据集，可以从以下链接下载：
- NEU-DET: http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/
- GC10-DET: https://www.kaggle.com/c/severstal-steel-defect-detection

### 模型训练

使用联合数据集训练模型：
```bash
python train.py
```

训练参数可以在`train.py`文件中修改。

### 模型推理

使用训练好的模型进行推理：
```bash
# 单张图片推理
python detect.py --source path/to/image.jpg --weights runs/detect/train/weights/best.pt

# 视频推理
python detect.py --source path/to/video.mp4 --weights runs/detect/train/weights/best.pt
```

## 性能指标

在联合数据集上的性能：
- mAP50: 待测试
- mAP50-95: 待测试

各类别性能：
- 类别1 (轧制鳞片): 待测试
- 类别2 (斑块): 待测试
- ...

## 许可证

本项目采用MIT许可证。

## 关于我们

**多焕智能（DooHuan AI）** 是一家专注于工业视觉检测和人工智能解决方案的公司，致力于为制造业提供高精度、高效率的智能检测系统。

## 联系方式

- **公司官网**：[多焕智能官网](https://www.doohuan.com)
- **GitHub**：[doohuan-ai](https://github.com/doohuan-ai)
- **邮箱**：reef@doohuan.com

## 致谢

- 感谢东北大学提供NEU-DET数据集
- 感谢GC10-DET数据集的贡献者
- 感谢Ultralytics提供YOLOv8开源实现 