# IMU数据处理与动作重建脚本

## 完整流程

### 1. extract_odt_data.py
从ODT文件提取IMU数据到TXT格式
- 输入: `data/m*.odt`
- 输出: `mydata/m*.txt`

### 2. process_new_imu_data.py
处理IMU数据
- TXT → CSV转换
- 用3号IMU数据填充损坏的0号IMU
- 坐标变换: (X上Y左Z前) → (X左Y上Z前)
- 输出: `mydata/processed_final/m*_final.csv`

### 3. infer_new_data.py
使用FIP模型进行动作重建推理
- 输入: CSV数据
- 输出: `mydata/processed_final/m*_poses.pt`
- **需要**: `ckpt/best_model.pt` 模型文件

### 4. apply_tpose_calibration.py
T-pose校准（重要！）
- m1强制设置为标准T-pose（所有关节为单位矩阵）
- m2-m7基于m1进行校准
- 输出: `mydata/processed_final/m*_poses_calibrated.pt`

### 5. visualize_smpl.py
渲染SMPL人体动画（Mac版本）
- 使用matplotlib渲染，无需OpenGL
- 人体正面朝向观察者
- 输出: `mydata/processed_final/m*_animation.mp4`
- **需要**: d2l conda环境

### 6. visualize_smpl_linux.py
渲染SMPL人体动画（Linux服务器版本）
- 使用pyrender + EGL离屏渲染
- 适用于无显示器的服务器环境

## 运行顺序

```bash
# 步骤1-3: 数据提取和推理
python scripts/extract_odt_data.py
python scripts/process_new_imu_data.py
python scripts/infer_new_data.py

# 步骤4: T-pose校准（必须！）
python scripts/apply_tpose_calibration.py

# 步骤5: 渲染视频
# Mac:
conda run -n d2l python scripts/visualize_smpl.py

# Linux:
python scripts/visualize_smpl_linux.py
```

## 重要说明

- **m1是T-pose校准数据**: 所有动作都基于m1的标准T-pose进行校准
- **校准数据独立存储**: `*_poses_calibrated.pt` 包含校准后的姿态
- **视频朝向**: 人体正面朝向观察者（Y轴方向）
- **坐标系**: X左，Y前，Z上

## 输出文件

```
mydata/
├── m*.txt                      # 提取的原始数据
├── csv/m*.csv                  # CSV格式
└── processed_final/
    ├── m*_final.csv            # 处理后的IMU数据
    ├── m*_poses.pt             # 推理结果
    ├── m*_poses_calibrated.pt  # 校准后的姿态
    └── m*_animation.mp4        # 最终视频
```
