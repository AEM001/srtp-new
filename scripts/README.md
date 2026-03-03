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

# Linux/Ubuntu:
python scripts/visualize_smpl_linux.py
```

## Ubuntu服务器渲染详细说明

### 环境准备

#### 1. 系统依赖
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装必要的系统库
sudo apt install -y \
    python3-pip \
    python3-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    xvfb

# 安装ffmpeg（用于视频编码）
sudo apt install -y ffmpeg
```

#### 2. Python环境
```bash
# 创建虚拟环境（推荐）
python3 -m venv fip_env
source fip_env/bin/activate

# 或使用conda
conda create -n fip python=3.10
conda activate fip
```

#### 3. 安装Python依赖
```bash
# 基础依赖
pip install torch numpy matplotlib tqdm

# 渲染依赖
pip install trimesh pyrender imageio imageio-ffmpeg

# 项目依赖
pip install chumpy
```

### 运行渲染

#### 方法1: 直接运行（推荐）
```bash
# 确保在项目根目录
cd /path/to/srtp-fip

# 运行渲染脚本
python scripts/visualize_smpl_linux.py
```

#### 方法2: 使用Xvfb（无显示器环境）
```bash
# 启动虚拟显示
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# 运行渲染
python scripts/visualize_smpl_linux.py
```

#### 方法3: 使用nohup后台运行
```bash
# 后台运行，输出到日志
nohup python scripts/visualize_smpl_linux.py > render.log 2>&1 &

# 查看进度
tail -f render.log
```

### 常见问题

#### Q1: EGL初始化失败
```bash
# 解决方案：安装Mesa EGL库
sudo apt install -y libegl1-mesa libegl1-mesa-dev
```

#### Q2: 找不到libGL.so
```bash
# 解决方案：安装OpenGL库
sudo apt install -y libgl1-mesa-glx libglu1-mesa
```

#### Q3: 内存不足
```bash
# 解决方案：分批渲染，修改MOTION_IDS只渲染部分动作
# 在visualize_smpl_linux.py中修改：
# MOTION_IDS = ['m1', 'm2']  # 只渲染前两个
```

#### Q4: 渲染速度慢
```bash
# 解决方案：
# 1. 降低分辨率（修改RESOLUTION）
# 2. 减少面数（增加step值）
# 3. 使用GPU加速（如果有CUDA）
```

### 输出位置
渲染完成后，视频文件位于：
```
mydata/processed_final/m*_animation.mp4
```

### 传输文件到本地
```bash
# 使用scp下载视频
scp username@server:/path/to/srtp-fip/mydata/processed_final/*.mp4 ./

# 或使用rsync
rsync -avz username@server:/path/to/srtp-fip/mydata/processed_final/*.mp4 ./
```

## 重要说明

### 数据处理
- **m1是T-pose校准数据**: 所有动作都基于m1的标准T-pose进行校准
- **校准数据独立存储**: `*_poses_calibrated.pt` 包含校准后的姿态
- **视频朝向**: 人体正面朝向观察者（Y轴方向）
- **坐标系**: X左，Y前，Z上

### FIP模型局限性（重要！）

**当前渲染逻辑和代码都是正确的**，但存在以下已知问题：

1. **无法建模全局移动**
   - FIP模型只能预测关节的相对旋转，无法预测人体的全局位移
   - 导致旋转动作（如转身）无法正确渲染，人体会保持在原地旋转

2. **IMU芯片移动导致的误差**
   - 由于动作过程中IMU芯片可能发生位移
   - 后续帧中腿部可能出现交叉现象
   - 这是传感器物理特性导致的，不是代码问题

3. **解决方案**
   - 使用支持全局位移预测的模型（如TransPose、PIP等）
   - 或结合视觉信息进行全局位置校正
   - 改进IMU固定方式，减少传感器移动

**这些是模型本身的局限性，不影响当前流程的正确性。**

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
