# FIP 实时 IMU 动作重建与流媒体系统

基于 [FIP (Fast Inertial Pose)](https://doi.org/10.1038/s41467-024-46662-5) 的实时 IMU 人体动作重建系统。  
6 个 IMU 传感器数据通过网络实时发送 → FIP 逐帧推理 → SMPL 网格渲染 → MJPEG 视频流远程查看。

---

## 系统架构

```
IMU 传感器设备
    │  TCP:9000  JSON-lines 协议
    ▼
┌─────────────────────────────────────────────────────┐
│              stream_server.py                        │
│                                                      │
│  ┌──────────────┐   队列   ┌────────────────────┐   │
│  │ TCP 接收线程 │ ──────▶ │  处理线程           │   │
│  │ (每客户端)   │         │  FIP 推理           │   │
│  └──────────────┘         │  SMPL 渲染          │   │
│                           │  JPEG 编码          │   │
│                           └────────┬───────────┘   │
│                                    │ 最新帧         │
│  ┌──────────────────────────────────▼────────────┐  │
│  │            Flask HTTP 服务器                   │  │
│  │  /stream    – MJPEG 视频流                    │  │
│  │  /calibrate – T-pose 校准                     │  │
│  │  /reset     – 重置流水线                      │  │
│  │  /status    – JSON 状态                       │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
    │  HTTP:8080
    ▼
浏览器 / VLC / 任何 MJPEG 客户端
```

**关键设计**：IMU 数据直接进入 FIP 模型，无中间文件（无 ODT、无 CSV、无 .pt）。

---

## 项目结构

```
├── stream_server.py           # 实时服务器主入口（TCP + HTTP）
├── config.py                  # 统一配置（路径、端口、设备、参数）
│
├── pipeline/
│   ├── realtime.py            # 实时推理流水线（RealtimePipeline）
│   ├── renderer.py            # SMPL 网格渲染器（pyrender，支持 osmesa/EGL）
│   ├── inference.py           # 离线推理工具（FIP 模型加载等）
│   └── preprocess.py          # 离线预处理（ODT → CSV，仅离线模式使用）
│
├── model/
│   ├── net.py                 # FIP LSTM 在线推理模型
│   └── math/                  # 旋转表示转换、运动学
│
├── src/                       # SMPL 运动学 / 坐标转换工具
├── human_body_prior/          # 人体先验库
│
├── ckpt/best_model.pt         # FIP 预训练权重
├── data/
│   ├── SMPL_male.pkl          # SMPL 模型
│   └── raw/                   # 原始 ODT 文件（仅离线模式）
│
├── examples/
│   └── send_imu.py            # 模拟 IMU 客户端（测试用）
│
├── Dockerfile                 # Docker 镜像（CPU/osmesa，无需 GPU）
├── docker-compose.yml         # 标准部署（CPU）
├── docker-compose.gpu.yml     # GPU 加速部署（EGL）
└── run_pipeline.py            # 离线批处理流水线（ODT → 视频）
```

---

## 快速部署（Docker，推荐）

### 前提条件

- Docker 20.10+
- Docker Compose v2+
- `ckpt/best_model.pt` 和 `data/SMPL_male.pkl` 已就位

### 构建并启动

```bash
# 构建镜像并在后台启动
docker compose up -d --build

# 查看启动日志
docker compose logs -f
```

启动后：
- **IMU 数据端口**: `TCP 9000`（发送 IMU 数据到此端口）
- **视频流地址**: `http://<服务器IP>:8080/`（浏览器直接打开）

### 停止服务

```bash
docker compose down
```

### GPU 加速（可选）

需要 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

---

## IMU 数据协议

客户端通过 **TCP 长连接** 连接到服务器 `9000` 端口，发送 **换行符分隔的 JSON**（JSON-lines 格式）。

### 数据帧格式

每个时间步发送一行 JSON，包含全部 6 个 IMU 的数据：

```json
{"t": 1234.567, "imus": [
  [ax, ay, az, roll, pitch, yaw],
  [ax, ay, az, roll, pitch, yaw],
  [ax, ay, az, roll, pitch, yaw],
  [ax, ay, az, roll, pitch, yaw],
  [ax, ay, az, roll, pitch, yaw],
  [ax, ay, az, roll, pitch, yaw]
]}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `t` | float | 时间戳（秒，任意基准） |
| `imus` | array[6][6] | 6 个 IMU 各 6 个浮点数 |
| `ax ay az` | float | 加速度（m/s²） |
| `roll pitch yaw` | float | 欧拉角（度，ZYX 顺序） |

**IMU 坐标系（原始传感器坐标）**：X-up · Y-left · Z-front  
服务器内部自动转换为模型坐标系（X-left · Y-up · Z-front）。

**IMU 顺序**（与离线 pipeline 一致）：
```
0: 骨盆/根  1: 左手腕  2: 右手腕
3: 左脚踝  4: 右脚踝  5: 头部
```

### 控制命令（同一 TCP 连接发送）

```json
{"cmd": "calibrate"}
{"cmd": "reset"}
```

---

## HTTP 接口

| 地址 | 说明 |
|------|------|
| `GET /` | 浏览器界面（内嵌视频流 + 控制按钮） |
| `GET /stream` | MJPEG 视频流（可在浏览器/VLC/ffplay 打开） |
| `GET /calibrate` | 将当前姿态设为 T-pose 参考 |
| `GET /reset` | 重置 LSTM 状态和 T-pose 校准 |
| `GET /status` | JSON 状态（帧率、是否已校准、帧数） |

**在任意浏览器打开**：
```
http://<服务器IP>:8080/
```

**VLC / ffplay 观看**：
```bash
ffplay http://<服务器IP>:8080/stream
vlc http://<服务器IP>:8080/stream
```

---

## T-pose 校准流程

1. 启动服务器（`docker compose up`）
2. 开始发送 IMU 数据（客户端连接 TCP:9000）
3. **受试者保持 T-pose 站立**（双臂水平展开，自然站立）
4. 触发校准（三选一）：
   - 浏览器点击「T-pose 校准」按钮
   - 访问 `http://<服务器IP>:8080/calibrate`
   - 通过 TCP 发送 `{"cmd": "calibrate"}\n`
5. 服务器响应 `{"ok": true, "message": "T-pose 已校准"}`，随后渲染姿态归零

---

## 测试（模拟 IMU 客户端）

项目内置模拟客户端，无需真实传感器即可测试：

```bash
# 本地测试（服务器在同一机器）
python examples/send_imu.py

# 连接远程服务器
python examples/send_imu.py --host 192.168.1.10 --port 9000 --fps 30

# 运行 60 秒后自动停止（2 秒时自动发送校准命令）
python examples/send_imu.py --host 10.0.0.5 --duration 60 --calibrate-at 2.0
```

---

## 配置参数

所有参数在 `config.py` 中定义，**均可通过环境变量覆盖**（Docker 部署时在 `docker-compose.yml` 的 `environment` 中修改）：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `IMU_PORT` | `9000` | IMU 数据 TCP 端口 |
| `STREAM_PORT` | `8080` | HTTP 视频流端口 |
| `PYOPENGL_PLATFORM` | `osmesa` | 渲染后端：`osmesa`（CPU）或 `egl`（GPU） |
| `STREAM_JPEG_QUALITY` | `85` | JPEG 质量（1-100，越高画质越好但带宽越大） |

也可以通过命令行参数覆盖：

```bash
python stream_server.py \
    --imu-port 9001 \
    --stream-port 8081 \
    --width 1280 --height 720 \
    --quality 90
```

---

## 技术细节

### 实时推理流水线

每帧处理步骤（全在内存中，无磁盘 I/O）：

```
IMU JSON 帧
  → 坐标变换（IMU 坐标 → 模型坐标）
  → 根坐标系相对化（以第 6 个 IMU 为根）
  → FIP LSTM 单步推理（维护隐藏状态跨帧）
  → T-pose 校准（R_cal = R_tpose^T × R_raw）
  → glb2local（全局旋转 → SMPL 局部旋转）
  → SMPL 前向运动学（关节位置 + 顶点）
  → pyrender 渲染（SMPL 网格 → RGB 帧）
  → JPEG 编码
  → MJPEG 推送
```

### 模型架构

- **FIP**: LSTM 在线推理，逐帧处理 6 个 IMU 的加速度和方向
- **输入**: 根坐标系相对加速度 `[18]` + 方向矩阵 `[54]` + 身体参数 `[4]`
- **输出**: 15 个关节的全局旋转矩阵 `[15, 3, 3]`

### 坐标系统

| 坐标系 | 朝向 |
|--------|------|
| IMU 原始 | X-up · Y-left · Z-front |
| 模型内部 | X-left · Y-up · Z-front |
| SMPL | 24 关节（FIP 预测 15 个，其余固定为单位旋转） |

### 渲染后端

| 后端 | 环境变量 | 适用场景 |
|------|----------|---------|
| osmesa | `PYOPENGL_PLATFORM=osmesa` | Docker CPU，无需任何 GPU/显示 |
| EGL | `PYOPENGL_PLATFORM=egl` | GPU 直通容器，速度更快 |

---

## 离线批处理模式（保留）

仍然支持从 ODT 文件离线处理：

```bash
# 完整离线流水线：ODT → CSV → 推理 → 渲染视频
python run_pipeline.py

# 分步执行
python run_pipeline.py --step preprocess    # ODT → CSV
python run_pipeline.py --step infer         # CSV → 校准姿态
python run_pipeline.py --step render        # 姿态 → MP4 视频

# 导出 SMPL 参数
python export_smpl_params.py --formats pose_aa vertices
```

---

## 已知限制

1. **全局位移**: FIP 仅预测关节旋转，根关节固定在原点（无全局平移）
2. **传感器漂移**: 长时间运动累积 IMU 误差时可重新校准
3. **numpy 版本**: 需要 `numpy<2` 以兼容 scipy/SMPL 模型加载
4. **osmesa 性能**: 软件渲染速度约 10-20 fps（取决于 CPU）；GPU EGL 可达 30+ fps
