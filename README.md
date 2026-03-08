# FIP Motion Reconstruction Pipeline

基于 [FIP (Fast Inertial Pose)](https://doi.org/10.1038/s41467-024-46662-5) 的 IMU 人体动作重建系统。  
从 6 个 IMU 传感器数据重建完整 SMPL 人体姿态，支持视频渲染和 SMPL 参数导出。

## 项目结构

```
├── config.py                  # 统一配置（路径、设备、参数）
├── run_pipeline.py            # 主流水线：预处理 → 推理 → 渲染
├── export_smpl_params.py      # SMPL 参数导出工具
├── requirements.txt
│
├── pipeline/                  # 核心流水线模块
│   ├── preprocess.py          #   ODT → 处理后 CSV
│   ├── inference.py           #   CSV → FIP 推理 + T-pose 校准
│   └── renderer.py            #   姿态 → SMPL 网格 → MP4 视频
│
├── model/                     # FIP 神经网络
│   ├── net.py                 #   LSTM 在线推理模型
│   └── math/                  #   旋转表示转换、运动学
├── src/                       # SMPL 运动学 / 评估工具
│   ├── kinematic_model.py     #   SMPL 前向运动学
│   ├── eval_tools.py          #   全局/局部坐标转换
│   └── math.py                #   数学工具
├── human_body_prior/          # 人体先验库（VPoser 等）
│
├── ckpt/best_model.pt         # FIP 预训练权重
├── data/
│   ├── SMPL_male.pkl          # SMPL 男性模型
│   └── raw/m*.odt             # 原始 IMU 录制（ODT 格式）
│
└── output/                    # 所有输出
    ├── csv/                   #   预处理后 IMU 数据
    ├── poses/                 #   校准后姿态（.pt 文件）
    ├── smpl_params/           #   导出的 SMPL 参数（.npz）
    └── video/                 #   渲染视频（.mp4）
```

## 快速开始

### 环境配置

```bash
# 推荐使用 conda（需要 numpy<2 以兼容 scipy/trimesh）
conda activate your_env
pip install -r requirements.txt
```

**依赖**: PyTorch, NumPy, pandas, pyrender, trimesh, imageio, opencv-python

### 主流水线

```bash
# 完整流水线：预处理 → 推理 → 渲染
python run_pipeline.py

# 分步执行
python run_pipeline.py --step preprocess    # 仅预处理
python run_pipeline.py --step infer         # 仅推理+校准
python run_pipeline.py --step render        # 仅渲染

# 处理指定动作
python run_pipeline.py --motions m2 m3
python run_pipeline.py --step render --motions m2
```

### SMPL 参数导出

```bash
# 导出所有动作的 SMPL 参数（默认：pose_aa + pose_rotmat + joints）
python export_smpl_params.py

# 导出指定动作
python export_smpl_params.py --motions m2 m3

# 选择导出格式
python export_smpl_params.py --formats pose_aa vertices

# 查看所有可用格式
python export_smpl_params.py --list-formats

# 自定义输出目录
python export_smpl_params.py --output_dir /path/to/output
```

**可用导出格式**:
- `pose_aa`: SMPL 局部姿态（axis-angle）`[T, 24, 3]`
- `pose_rotmat`: SMPL 局部姿态（旋转矩阵）`[T, 24, 3, 3]`
- `pose_global`: 15 关节全局旋转（校准后）`[T, 15, 3, 3]`
- `joints`: 关节位置（前向运动学）`[T, 24, 3]`
- `vertices`: 网格顶点位置 `[T, 6890, 3]`

## 流水线说明

### 完整流程

| 步骤 | 模块 | 输入 → 输出 | 说明 |
|------|------|------------|------|
| **1. 预处理** | `pipeline/preprocess.py` | `data/raw/m*.odt` → `output/csv/m*_processed.csv` | 解析 ODT，修复损坏 IMU，坐标变换 |
| **2. 推理** | `pipeline/inference.py` | CSV → `output/poses/m*_calibrated.pt` | FIP 在线推理 + T-pose 校准 |
| **3. 渲染** | `pipeline/renderer.py` | poses.pt → `output/video/m*_animation.mp4` | SMPL 网格生成 + pyrender 渲染 |
| **4. 导出** | `export_smpl_params.py` | poses.pt → `output/smpl_params/m*_smpl.npz` | 标准 SMPL 参数导出 |

### T-pose 校准

- **m1** 是 T-pose 校准录制，所有帧强制为标准 T-pose（单位旋转）
- **m2-m7** 使用 m1 的首帧作为参考进行 T-pose 校准：`R_calibrated = R_tpose^(-1) @ R_motion`

### 数据格式

**校准后姿态文件** (`output/poses/m*_calibrated.pt`):
```python
{
    'poses': torch.Tensor,      # [T, 15, 3, 3] 全局旋转矩阵
    'num_frames': int,
    'calibrated': True
}
```

**导出 SMPL 参数** (`output/smpl_params/m*_smpl.npz`):
```python
{
    'pose_aa': np.ndarray,      # [T, 24, 3] 可选
    'pose_rotmat': np.ndarray,  # [T, 24, 3, 3] 可选
    'joints': np.ndarray,       # [T, 24, 3] 可选
    'vertices': np.ndarray,     # [T, 6890, 3] 可选
    'num_frames': int
}
```

## 技术细节

### 模型架构
- **FIP**: LSTM 在线推理，逐帧处理 6 个 IMU 的加速度和方向
- **输入**: 根坐标系相对加速度 `[18]` + 方向矩阵 `[54]` + 身体参数 `[4]`
- **输出**: 15 个关节的全局旋转矩阵 `[15, 3, 3]`

### 坐标系统
- **IMU 坐标系**: X-up, Y-left, Z-front
- **模型坐标系**: X-left, Y-up, Z-front
- **SMPL**: 24 关节（FIP 预测 15 个，其余固定为单位旋转）

### 渲染配置
- **引擎**: pyrender + EGL 离屏渲染（Linux 无显示器环境）
- **分辨率**: 640×480 @ 30fps（可在 `config.py` 修改）
- **相机**: 透视投影，距离 2.5m

## 已知问题

1. **环境依赖**: 需要 `numpy<2` 以兼容旧版 scipy（SMPL 模型加载）
2. **全局位移**: FIP 仅预测关节旋转，无全局平移（根关节固定在原点）
3. **传感器漂移**: 长时间录制可能累积 IMU 误差

## 引用

```bibtex
@article{yi2024fip,
  title={Fast Inertial Poser: Sparse Inertial Sensing for Real-time Human Motion Reconstruction},
  author={Yi, Xinyu and Zhou, Yuxiao and Xu, Feng},
  journal={Nature Communications},
  year={2024}
}
```
