# 新IMU数据处理快速指南

## 数据配置
- **IMU位置**: 0=头(坏,用3填充) 1=左手 2=右手 3=根节点 4=左腿 5=右腿
- **坐标系**: 原始(X上Y左Z前) → 目标(X左Y上Z前)
- **数据**: m1=T-pose校准, m2-m7=动作序列

## 运行流程

### 1. 提取数据
```bash
python scripts/extract_odt_data.py
```
输出: `mydata/m*.txt`

### 2. 处理数据
```bash
python scripts/process_new_imu_data.py
```
输出: `mydata/processed_final/m*_final.csv`

### 3. 动作重建
```bash
python scripts/infer_new_data.py
```
输出: `mydata/processed_final/m*_poses.pt`

**注意**: 需先下载模型到 `ckpt/best_model.pt`
- 链接: https://drive.google.com/drive/folders/1rxYTv5j8G-10Sxy1gwmZrP-NUHJPVqyo

### 4. 可视化
```bash
python scripts/visualize_new_data.py
```
输出: `output_m*.mp4`

## 目录结构
```
scripts/          # 新数据处理脚本
archive/          # 旧代码归档
mydata/           # 数据文件
  ├── m*.txt      # 提取的原始数据
  ├── csv/        # CSV格式
  └── processed_final/  # 处理后数据和结果
```

## 原DIP-IMU数据集
使用原始脚本: `preprocess.py`, `evaluate.py`, `visualize.py`
