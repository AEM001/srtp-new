# FIP Motion Reconstruction Pipeline

基于 [FIP (Fast Inertial Pose)](https://doi.org/10.1038/s41467-024-46662-5) 的 IMU 人体动作重建系统。  
从 6 个 IMU 传感器数据重建完整 SMPL 人体姿态并渲染为视频。

## 项目结构

```
├── config.py                  # 统一配置
├── run_pipeline.py            # 流水线入口（离线模式）
├── requirements.txt
│
├── pipeline/                  # 核心流水线模块
│   ├── preprocess.py          #   ODT → 处理后 CSV
│   ├── inference.py           #   CSV → FIP推理 + T-pose校准 → poses.pt
│   └── renderer.py            #   poses.pt → MP4 视频 (Linux EGL)
│
├── model/                     # FIP 神经网络
│   ├── net.py
│   └── math/
├── src/                       # SMPL 运动学 / 数学工具
├── human_body_prior/          # 人体先验库
│
├── ckpt/best_model.pt         # 预训练权重
├── data/
│   ├── SMPL_male.pkl          # SMPL 模型
│   └── raw/m*.odt             # 原始 IMU 录制
│
└── output/                    # 所有输出
    ├── csv/                   #   处理后 IMU CSV
    ├── poses/                 #   校准后姿态 .pt
    └── video/                 #   渲染视频 .mp4
```

## 快速开始

### 环境

```bash
# 推荐使用 conda (需要 numpy<2 以兼容 scipy/trimesh)
conda activate d2l   # 或你自己的环境
pip install -r requirements.txt
```

### 运行

```bash
# 完整流水线: ODT → CSV → 推理 → 渲染
python run_pipeline.py

# 仅渲染（使用已有的 poses）
python run_pipeline.py --step render

# 仅推理 + 校准
python run_pipeline.py --step infer

# 处理指定动作
python run_pipeline.py --motions m2 m3

# 仅渲染 m2
python run_pipeline.py --step render --motions m2
```

### 输出

渲染完成后视频位于 `output/video/m*_animation.mp4`。

## 流水线说明

| 步骤 | 模块 | 输入 → 输出 |
|------|------|------------|
| 1. 预处理 | `pipeline/preprocess.py` | `data/raw/m*.odt` → `output/csv/m*_processed.csv` |
| 2. 推理+校准 | `pipeline/inference.py` | CSV → `output/poses/m*_calibrated.pt` |
| 3. 渲染 | `pipeline/renderer.py` | poses.pt → `output/video/m*_animation.mp4` |

- **m1** 是 T-pose 校准录制，强制输出为标准 T-pose（单位矩阵）
- **m2-m7** 基于 m1 进行 T-pose 校准后渲染

## 技术细节

- **模型**: FIP (Fast Inertial Pose)，LSTM 在线推理，支持逐帧处理
- **渲染**: pyrender + EGL 离屏渲染（Linux 无显示器环境）
- **SMPL**: 24 关节参数化人体模型，15 个 FIP 预测关节

### 已知局限

1. FIP 仅预测关节相对旋转，无法建模全局位移
2. IMU 传感器物理移动可能导致后续帧误差累积
