# FIP 实时 IMU 动作重建与流媒体系统

基于 [FIP (Fast Inertial Pose)](https://doi.org/10.1038/s41467-024-46662-5) 的实时 IMU 人体动作重建系统。  
多个 ESP32 IMU 节点通过公网 frp 隧道发送数据 → tcp_aggregator 组帧 → FIP 逐帧推理 → SMPL 网格渲染 → MJPEG 视频流实时查看。

---

## 系统架构

```
ESP32 节点 ×N（远程）
    │  WiFi → TCP → 49.234.57.210:8001（frp 公网端口）
    │  数据格式：{"node":0,"t":12.3,"acc":[ax,ay,az],"rpy":[r,p,y]}\n
    ▼
┌──────────────────────────────────────────┐
│  frp 服务端 (49.234.57.210)              │
│  remotePort=8001 → localPort=9001 (TCP)  │
└──────────────────────────────────────────┘
    │  转发到本机 127.0.0.1:9001
    ▼
┌──────────────────────────────────────────────────────────────┐
│  tcp_aggregator.py  (本机)                                   │
│                                                              │
│  • 监听 0.0.0.0:9001  接收各 ESP32 节点的 per-node JSON     │
│  • yaw 清零（防止陀螺仪积分漂移）                            │
│  • 低通滤波 roll/pitch（alpha=0.2，平滑噪声）                │
│  • 以 20Hz 组装 6-IMU 帧转发到 stream_server:9000           │
│  • 同时广播到 monitor 端口 127.0.0.1:9002                   │
└──────────────────────────────────────────────────────────────┘
    │  TCP:9000  {"t":...,"imus":[[ax,ay,az,r,p,y]×6]}\n
    ▼
┌──────────────────────────────────────────────────────────────┐
│  stream_server.py  (本机)                                    │
│                                                              │
│  TCP:9000 接收 ──→ 队列 ──→ 处理线程（EGL 上下文所在线程）  │
│                              ↓                               │
│                         FIP LSTM 推理                        │
│                         SMPL 网格渲染（pyrender + EGL）      │
│                         JPEG 编码                            │
│                              ↓                               │
│                         Flask HTTP 服务器                    │
│  GET /          浏览器界面（内嵌 MJPEG）                     │
│  GET /stream    MJPEG 流                                     │
│  GET /calibrate T-pose 校准                                  │
│  GET /reset     重置流水线                                   │
│  GET /status    JSON 状态                                    │
└──────────────────────────────────────────────────────────────┘
    │  HTTP:8080
    ▼
浏览器 / VLC / 任何 MJPEG 客户端
```

**关键设计**：IMU 数据直接进入 FIP 模型，无中间文件（无 ODT、无 CSV、无 .pt）。

---

## 端口一览

| 端口 | 协议 | 用途 |
|------|------|------|
| `9001` | TCP | tcp_aggregator 监听 ESP32 连接（frp 转发） |
| `9000` | TCP | stream_server 接收组帧数据（来自 aggregator） |
| `9002` | TCP | aggregator monitor 广播（供 imu_monitor.py 连接） |
| `8080` | HTTP | stream_server 对外提供 MJPEG 流和浏览器界面 |
| `8001` | TCP | frp 公网端口（49.234.57.210），ESP32 连接此地址 |

---

## 项目结构

```
├── stream_server.py           # 实时服务器主入口（TCP:9000 → HTTP:8080）
├── tcp_aggregator.py          # ESP32 节点聚合器（TCP:9001 → TCP:9000）
├── config.py                  # 统一配置（路径、端口、设备、参数）
│
├── pipeline/
│   ├── realtime.py            # 实时推理流水线（RealtimePipeline）
│   │                          # ⚠ renderer 在处理线程中延迟初始化（EGL 线程亲和性）
│   ├── renderer.py            # SMPL 网格渲染器（pyrender + EGL/osmesa）
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
├── tools/
│   └── imu_monitor.py         # 实时 IMU 稳定性监控终端工具
│
├── examples/
│   ├── send_imu.py            # 模拟 6-IMU 组帧客户端（直连 stream_server，测试用）
│   ├── esp32_imu_tcp.ino      # ESP32 参考固件
│   └── esp32_tcp_final.ino    # ESP32 实际烧录固件（远程设备使用）
│
├── Dockerfile                 # Docker 镜像（CPU/osmesa，无需 GPU）
├── docker-compose.yml         # 标准部署（CPU）
├── docker-compose.gpu.yml     # GPU 加速部署（EGL）
└── run_pipeline.py            # 离线批处理流水线（ODT → 视频）
```

---

## 快速启动（本机裸机运行）

> 当前实际部署方式，不使用 Docker。

### 前提条件

- Python 虚拟环境：`/home/albert/learn/l-vllm/.venv`
- `ckpt/best_model.pt` 和 `data/SMPL_male.pkl` 已就位
- frp 客户端（`frpc`）已配置并运行

### 第一步：启动 stream_server

```bash
cd /home/albert/code/srtp/srtp-new
PYOPENGL_PLATFORM=egl \
  /home/albert/learn/l-vllm/.venv/bin/python stream_server.py \
  > /tmp/stream_server.log 2>&1 &
```

> **注意**：必须设置 `PYOPENGL_PLATFORM=egl`（当前环境 osmesa 不可用）。  
> 启动需约 15-20 秒加载模型。查看启动状态：
> ```bash
> tail -f /tmp/stream_server.log
> ```
> 看到 `Renderer initialized in processing thread.` 表示就绪。

### 第二步：启动 tcp_aggregator

```bash
cd /home/albert/code/srtp/srtp-new
python3 tcp_aggregator.py > /tmp/aggregator.log 2>&1 &
```

查看状态：
```bash
tail -f /tmp/aggregator.log
```
看到 `Monitor broadcast listening on 127.0.0.1:9002` 表示就绪。

### 第三步：启动 frpc（frp 客户端）

确认 frpc 已运行（将远程公网端口 8001 映射到本机 9001）：

```bash
# 检查 frpc 是否运行
pgrep -a frpc

# 若未运行，手动启动（根据实际 frpc 配置文件路径）
frpc -c /path/to/frpc.toml &
```

frpc 配置示例（`frpc.toml`）：
```toml
serverAddr = "49.234.57.210"
serverPort = 7000

[[proxies]]
name       = "imu-tcp"
type       = "tcp"
localIP    = "127.0.0.1"
localPort  = 9001
remotePort = 8001
```

### 第四步：验证运行状态

```bash
# 检查 stream_server 是否在处理帧
curl http://localhost:8080/status
# 期望输出：{"calibrated":false,"fps":20.0,"frames":1234,...}

# 检查 aggregator 活跃节点
grep "Active nodes" /tmp/aggregator.log | tail -3
# 期望输出：Active nodes: [0, 1, 2, 3, 5]
```

### 第五步：打开浏览器

```
http://localhost:8080/
```

---

## ESP32 固件说明

烧录到远程 ESP32 的是 `examples/esp32_tcp_final.ino`：

```cpp
// 关键配置
#define NODE_INDEX  0               // 节点编号 0-5，每块板不同
#define SERVER_IP   "49.234.57.210" // frp 服务器公网 IP
#define SERVER_PORT 8001            // frp 公网端口（映射到本机 9001）
```

**ESP32 发送格式**（每帧一行 JSON）：
```json
{"node":0,"t":12.345,"acc":[ax,ay,az],"rpy":[roll,pitch,yaw]}
```

| 字段 | 说明 |
|------|------|
| `node` | 节点编号 0-5 |
| `t` | `millis()/1000.0`（ESP32 启动后秒数） |
| `acc` | 加速度 m/s²，`[x, y, z]` |
| `rpy` | 互补滤波器输出欧拉角（度）`[roll, pitch, yaw]` |

> **Yaw 漂移说明**：ESP32 的 yaw 为纯陀螺仪积分，无磁力计校正，长时间会累积漂移。  
> tcp_aggregator 已在接收时**强制将 yaw 置为 0**，仅使用有加速度计校正的 roll/pitch 进行渲染。

**IMU 节点对应身体部位**：
```
节点 0: 骨盆（根节点）
节点 1: 左手腕
节点 2: 右手腕
节点 3: 左脚踝
节点 4: 右脚踝
节点 5: 头部
```

---

## tcp_aggregator 说明

`tcp_aggregator.py` 是 ESP32 节点与 stream_server 之间的聚合中间件。

### 功能

1. **多节点接收**：监听 `0.0.0.0:9001`，每个 ESP32 独立建立 TCP 连接
2. **数据预处理**：
   - `yaw = 0.0`：强制清除 yaw（防陀螺仪积分漂移导致渲染抖动）
   - 低通滤波：`rpy_smooth = 0.2 × rpy_raw + 0.8 × rpy_prev`（平滑噪声）
3. **组帧转发**：以 20Hz 将最新的各节点数据组装为 6-IMU 帧发送到 `stream_server:9000`
4. **数据时效性**：超过 0.5 秒未更新的节点数据视为过期，该槽位填充零值
5. **Monitor 广播**：每帧同时广播到 `127.0.0.1:9002`，供监控工具连接（含各节点 `ages_ms` 延迟信息）

### 关键参数（文件顶部修改）

```python
FRAME_HZ     = 20    # 组帧发送频率（Hz）
STALE_S      = 0.5   # 节点数据过期阈值（秒）
SMOOTH_ALPHA = 0.2   # 低通滤波系数（越小越平滑，响应越慢）
```

### 多节点时间戳同步策略

**当前策略：最新数据快照（Latest-Value Snapshot）**

每次组帧时，直接取各节点"最新收到的那帧"数据，不做跨节点的时间戳对齐。

**为什么不做严格对齐？**

| 因素 | 说明 |
|------|------|
| FIP 模型特性 | LSTM 是时序鲁棒模型，训练时本身包含数十毫秒的同步误差，对节点间 <100ms 偏差不敏感 |
| 网络延迟分布 | 各节点经 frp 隧道传输，典型延迟 20-80ms，不同节点差异通常 <50ms |
| STALE_S 保障 | 超过 500ms 未更新的节点数据会被丢弃（填零），防止使用极度过期数据 |

**何时偏差会有问题？**

若某节点因 WiFi 不稳定导致数据延迟 >100ms（比其他节点晚 2 个以上 ESP32 帧），该节点姿态数据与其他节点不同步，可能引起对应肢体轻微抖动。这种情况可通过 `imu_monitor.py` 的**延迟列**实时观察。

**如何排查**：

```bash
python tools/imu_monitor.py
# 查看每行右侧的"延迟"列（单位 ms）
# 若某节点延迟比其他节点高出 >100ms，标注 ⏱滞后
# 顶部出现 "⚠ 节点间时间戳偏差 XXXms" 警告时，说明节点间同步较差
```

---

## stream_server 说明

`stream_server.py` 接收组帧数据，执行 FIP 推理 + SMPL 渲染，对外提供 MJPEG 流。

### 重要实现细节

**EGL 线程亲和性**：pyrender 的 EGL 上下文必须在创建它的线程中使用。  
因此 `SMPLRenderer` 在 `_processing_loop` 线程启动时延迟初始化（`pipeline.init_renderer()`），而非在主线程构造。

### HTTP 接口

| 地址 | 说明 |
|------|------|
| `GET /` | 浏览器界面（内嵌 MJPEG 流 + 控制按钮） |
| `GET /stream` | 原始 MJPEG 流（VLC/ffplay/浏览器直接打开） |
| `GET /calibrate` | 将当前姿态设为 T-pose 参考 |
| `GET /reset` | 重置 LSTM 隐藏状态和 T-pose 校准 |
| `GET /status` | JSON 状态，示例：`{"calibrated":true,"fps":20.5,"frames":1075}` |

### 命令行参数

```bash
python stream_server.py \
    --imu-port 9000 \       # IMU 数据 TCP 端口（默认 9000）
    --stream-port 8080 \    # HTTP 视频流端口（默认 8080）
    --width 640 \           # 渲染宽度（默认 640）
    --height 480 \          # 渲染高度（默认 480）
    --quality 85            # JPEG 质量（默认 85）
```

---

## T-pose 校准流程

1. 确认 stream_server 和 aggregator 正常运行，`/status` 显示 `fps > 0`
2. **受试者保持 T-pose 站立**（双臂水平展开，身体直立）
3. 触发校准（三选一）：
   - 浏览器打开 `http://localhost:8080/` 点击「T-pose 校准」按钮
   - 直接访问 `http://localhost:8080/calibrate`
   - TCP 发送：`echo '{"cmd":"calibrate"}' | nc 127.0.0.1 9000`
4. 返回 `{"ok": true}` 后渲染姿态自动归零

---

## IMU 稳定性监控

```bash
cd /home/albert/code/srtp/srtp-new
/home/albert/learn/l-vllm/.venv/bin/python tools/imu_monitor.py
```

连接到 `127.0.0.1:9002`（aggregator monitor 广播端口），实时显示每个节点的：
- 当前 roll / pitch / yaw 值
- 最近 30 帧的标准差（抖动量，`std > 3°` 为不稳定）
- 帧间跳变量（`delta > 5°` 为异常跳帧）

**输出示例**：
```
节点      roll   pitch   yaw  std_r  std_p  std_y  跳变_r  跳变_p  跳变_y  状态
骨盆     116.2   -73.2   0.0   0.66   0.93   0.00    0.02    0.07    0.00  ✓ 稳定
左手腕    92.9   -33.5   0.0   1.07   0.73   0.00    0.06    0.10    0.00  ✓ 稳定
```

> **注意**：运行 monitor 前必须先启动 aggregator（提供 9002 端口）。

---

## 测试（模拟 IMU 客户端）

无需 ESP32 硬件，可用内置模拟客户端直接测试 stream_server：

```bash
# 模拟 6 路 IMU 数据，直连 stream_server:9000
/home/albert/learn/l-vllm/.venv/bin/python examples/send_imu.py

# 连接远程 stream_server
python examples/send_imu.py --host 192.168.1.10 --port 9000 --fps 30
```

> `send_imu.py` 直接发送组帧格式（`{"t":...,"imus":[...]}`），绕过 aggregator。

---

## 坐标系规范（ESP32 团队必读）

### ESP32 发送坐标系（传感器本地坐标）

**右手坐标系，定义如下**：

| 轴 | 正方向 | 说明 |
|----|--------|------|
| **X** | 向上 ↑ | 传感器垂直向上 |
| **Y** | 向左 ← | 传感器左侧 |
| **Z** | 向前 → | 传感器前方 |

**示意图**（传感器平放，Z 指向屏幕外）：
```
        X (up)
        ↑
        │
Y ← ──┼──
      │
     Z (front, out of screen)
```

### 数据格式

ESP32 每帧发送（JSON 一行）：
```json
{"node":0,"t":12.345,"acc":[ax,ay,az],"rpy":[roll,pitch,yaw]}
```

| 字段 | 单位 | 坐标轴定义 |
|------|------|-----------|
| `acc[0]` (ax) | m/s² | X-up 方向加速度 |
| `acc[1]` (ay) | m/s² | Y-left 方向加速度 |
| `acc[2]` (az) | m/s² | Z-front 方向加速度 |
| `rpy[0]` (roll) | 度 | 绕 X 轴旋转（右手定则） |
| `rpy[1]` (pitch) | 度 | 绕 Y 轴旋转（右手定则） |
| `rpy[2]` (yaw) | 度 | **已忽略**，强制置 0 |

> **重要**：yaw 会被 aggregator 强制清零（`rpy[2] = 0`），因为纯陀螺仪积分会无限漂移。渲染只依赖 roll/pitch。

### 传感器安装方向

**6 个传感器安装位置**：

| 节点 | 身体部位 | 建议安装方向 |
|------|---------|-------------|
| 0 | 骨盆/腰部 | Z 向前（面朝方向），X 向上 |
| 1 | 左手腕 | Z 向前（手背方向），X 向上 |
| 2 | 右手腕 | Z 向前（手背方向），X 向上 |
| 3 | 左脚踝 | Z 向前（脚背方向），X 向上 |
| 4 | 右脚踝 | Z 向前（脚背方向），X 向上 |
| 5 | 头部 | Z 向前（面朝方向），X 向上 |

**校验方法**：传感器静止平放时（Z 向上），应输出 `acc ≈ [0, 0, 9.8]`。

### 内部坐标转换（自动完成）

stream_server 收到数据后，自动应用以下转换（**无需 ESP32 端修改**）：

```python
# 转换矩阵：IMU (X-up,Y-left,Z-front) → Model (X-left,Y-up,Z-front)
T = [[0,1,0],
     [1,0,0],
     [0,0,1]]

acc_model = T @ acc_imu    # 即：[ay, ax, az]
R_model   = T @ R_imu @ T.T
```

这意味着：
- 你的 X-up 变成模型的 Y-up
- 你的 Y-left 变成模型的 X-left  
- Z 保持向前

### T-pose 校准姿势

校准时受试者应站立：
- **双臂水平侧平举**（与肩膀同高，手掌向下）
- **身体直立**，面朝前方
- **双腿并拢**或稍微分开

此时所有传感器应大致处于同一竖直平面内（冠状面）。

---

## 技术细节

### 实时推理流水线（每帧）

```
TCP 接收 JSON 帧 {"t":..., "imus":[[ax,ay,az,r,p,y]×6]}
  → 坐标变换（IMU 坐标 → 模型坐标，_COORD_TRANSFORM 矩阵）
  → 根坐标系相对化（以节点 5/头部 为根）
  → FIP LSTM 单步推理（跨帧维护隐藏状态 integ_hc / hip_hc / spine_hc）
  → T-pose 校准（R_cal = R_tpose^T × R_raw，未校准时跳过）
  → glb2local（全局旋转 → SMPL 局部旋转）
  → SMPL 前向运动学（24 关节，FIP 预测 15 个，其余固定为单位旋转）
  → pyrender 渲染（EGL 离屏渲染 → RGB 帧）
  → JPEG 编码（cv2.imencode）
  → MJPEG 推送
```

### 数据流格式对比

| 位置 | 格式 | 说明 |
|------|------|------|
| ESP32 → aggregator | `{"node":N,"t":T,"acc":[...],"rpy":[...]}` | per-node，每节点独立连接 |
| aggregator → stream_server | `{"t":T,"imus":[[×6]×6]}` | 组帧，20Hz |
| aggregator → monitor | 同上 | 广播副本 |

### 渲染后端

| 后端 | 环境变量 | 当前状态 |
|------|----------|---------|
| EGL | `PYOPENGL_PLATFORM=egl` | ✅ **当前使用**，需要 GPU |
| osmesa | `PYOPENGL_PLATFORM=osmesa` | ❌ 当前环境不可用（`OSMesaCreateContextAttribs` 缺失） |

---

## 故障排查

### stream_server 显示 "Waiting for IMU data"

```bash
# 检查帧数
curl http://localhost:8080/status

# 检查 aggregator 是否在发帧
grep "Sending frame" /tmp/aggregator.log | tail -3

# 检查 9000 端口连接
ss -tnp | grep 9000
```

### stream_server 渲染报错 `eglMakeCurrent failed`

EGL context 线程亲和性问题。确认 `pipeline/realtime.py` 中 `renderer = None`（构造时不初始化），由 `_processing_loop` 调用 `pipeline.init_renderer()` 完成初始化。

### 渲染人形抖动不稳定

```bash
# 运行实时监控，查看哪些节点数据抖动
python tools/imu_monitor.py
```

- **std > 3°**：传感器噪声，检查 ESP32 固件和传感器安装
- **yaw 漂移**：aggregator 已自动清零，无需处理
- 如需调整平滑强度，修改 `tcp_aggregator.py` 中 `SMOOTH_ALPHA`（降低 = 更平滑但响应更慢）

### aggregator 显示节点数据过期

```bash
grep "stale" /tmp/aggregator.log | tail -5
```

检查 ESP32 发送频率是否低于 `1/STALE_S = 2Hz`，或网络延迟是否过高。

### frpc 未运行

```bash
pgrep -a frpc || echo "frpc 未运行！"
# ESP32 将无法连接到本机
```

---

## 已知限制

1. **全局位移**：FIP 仅预测关节旋转，根关节固定在原点（无全局平移）
2. **Yaw 固定为零**：当前实现将所有节点 yaw 清零，人物朝向不随头部/身体转动变化
3. **右脚踝缺失**：如果 ESP32 节点 4 未连接，右脚踝槽位为零值，渲染中该肢体保持默认姿态
4. **numpy 版本**：需要 `numpy<2` 以兼容 scipy/SMPL 模型加载
5. **osmesa 不可用**：当前 Python 环境的 osmesa 绑定缺少 `OSMesaCreateContextAttribs`，只能使用 EGL
