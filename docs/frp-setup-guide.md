# FRP 内网穿透配置指南

> 本文档记录了当前项目的内网穿透配置，方便在其他场景复用。
> 
> 适用版本：frp v0.61.1
> 服务端：49.234.57.210 (腾讯云服务器)
> 客户端：本地 Ubuntu 开发机

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          公网服务器 (49.234.57.210)                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     frps (服务端)                                │   │
│  │                    端口: 7000 (控制端口)                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ 长连接保持                               │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │   8000  │   8001  │   8002  │   8003  │   8004  │   9000      │   │
│  │   SSH   │ IMU数据 │ HTTP流  │原始监控 │Web监控  │ 流服务器    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ 反向代理
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          本地开发机 (Ubuntu)                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     frpc (客户端)                                │   │
│  │                    配置文件: /etc/frp/frpc.toml                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │  :22     │  :9001   │  :8080   │  :9002   │  :9003   │  :9000   │   │
│  │  SSH     │aggregator│stream    │raw       │Web       │stream    │   │
│  │          │ESP32入口 │HTTP输出  │TCP监控   │WebSocket │TCP输入   │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 服务端配置 (frps)

### 2.1 基本信息

| 项目 | 值 |
|------|-----|
| 服务器地址 | `49.234.57.210` |
| 控制端口 | `7000` |
| 认证方式 | Token |
| 认证令牌 | `yunnan123456` |

### 2.2 服务端允许的端口范围

当前服务端配置允许的远程端口：
- `8000-8004` (项目专用)
- `9000` (项目专用)
- 其他端口需要服务端配置放行

> **注意**：如果需要在其他场景复用，请确保服务端 `frps.toml` 中配置了相应的 `allowPorts` 或关闭端口限制。

### 2.3 服务端配置参考 (frps.toml)

```toml
# 服务端配置文件参考（位于 49.234.57.210）
bindPort = 7000
auth.method = "token"
auth.token = "yunnan123456"

# 允许代理的端口范围（根据实际需求调整）
allowPorts = [
    { start = 8000, end = 8010 },
    { start = 9000, end = 9100 }
]
```

---

## 3. 客户端配置 (frpc)

### 3.1 配置文件路径

```
/etc/frp/frpc.toml
```

### 3.2 完整配置内容

```toml
serverAddr = "49.234.57.210"
serverPort = 7000
auth.method = "token"
auth.token = "yunnan123456"
 
[[proxies]]
name = "ssh"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = 8000
 
[[proxies]]
name = "stream-tcp"
type = "tcp"
localIP = "127.0.0.1"
localPort = 9000
remotePort = 9000
 
[[proxies]]
name = "imu-tcp"
type = "tcp"
localIP = "127.0.0.1"
localPort = 9001
remotePort = 8001
 
[[proxies]]
name = "stream-http"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8080
remotePort = 8002

[[proxies]]
name = "imu-monitor"
type = "tcp"
localIP = "127.0.0.1"
localPort = 9002
remotePort = 8003

[[proxies]]
name = "imu-web"
type = "tcp"
localIP = "127.0.0.1"
localPort = 9003
remotePort = 8004
```

### 3.3 端口映射详情表

| 代理名称 | 类型 | 本地服务 | 本地端口 | 远程端口 | 用途说明 |
|---------|------|---------|---------|---------|---------|
| `ssh` | TCP | SSH | 22 | 8000 | SSH远程登录开发机 |
| `stream-tcp` | TCP | stream_server TCP输入 | 9000 | 9000 | FIP推理服务TCP端口 |
| `imu-tcp` | TCP | tcp_aggregator | 9001 | 8001 | ESP32数据聚合入口 |
| `stream-http` | TCP | stream_server HTTP | 8080 | 8002 | MJPEG视频流输出 |
| `imu-monitor` | TCP | aggregator原始监控 | 9002 | 8003 | 原始TCP监控数据 |
| `imu-web` | TCP | aggregator Web监控 | 9003 | 8004 | WebSocket监控页面 |

---

## 4. 客户端服务管理

### 4.1 Systemd 服务配置

**文件路径**: `/etc/systemd/system/frpc.service`

```ini
[Unit]
Description=FRP Client (frpc)
After=network.target

[Service]
Type=simple
Restart=always
RestartSec=5
ExecStart=/usr/local/bin/frpc -c /etc/frp/frpc.toml

[Install]
WantedBy=multi-user.target
```

### 4.2 常用管理命令

```bash
# 启动服务
sudo systemctl start frpc

# 停止服务
sudo systemctl stop frpc

# 重启服务（修改配置后使用）
sudo systemctl restart frpc

# 查看状态
sudo systemctl status frpc

# 设置开机自启
sudo systemctl enable frpc

# 禁用开机自启
sudo systemctl disable frpc

# 查看实时日志
sudo journalctl -u frpc -f
```

### 4.3 检查服务运行状态

```bash
# 检查进程
pgrep -a frpc

# 检查端口监听
sudo ss -tlnp | grep frpc

# 检查连接状态
cat /tmp/frpc.log 2>/dev/null || sudo journalctl -u frpc --no-pager | tail -20
```

---

## 5. 快速复用指南

### 5.1 新机器部署步骤

假设你需要在新机器上部署相同的内网穿透配置：

```bash
# 1. 安装 frp（确保版本一致 v0.61.1）
# 下载对应架构的二进制文件
wget https://github.com/fatedier/frp/releases/download/v0.61.1/frp_0.61.1_linux_amd64.tar.gz
tar -xzf frp_0.61.1_linux_amd64.tar.gz
sudo cp frp_0.61.1_linux_amd64/frpc /usr/local/bin/
sudo chmod +x /usr/local/bin/frpc

# 2. 创建配置目录
sudo mkdir -p /etc/frp

# 3. 复制配置文件（使用本文档第3.2节的配置）
sudo tee /etc/frp/frpc.toml << 'EOF'
[粘贴上述完整配置内容]
EOF

# 4. 创建 systemd 服务
sudo tee /etc/systemd/system/frpc.service << 'EOF'
[粘贴第4.1节的配置]
EOF

# 5. 启动并设置开机自启
sudo systemctl daemon-reload
sudo systemctl enable frpc
sudo systemctl start frpc

# 6. 验证
sudo systemctl status frpc
```

### 5.2 新增端口映射（示例）

以添加新的 Web 服务为例：

```bash
# 1. 编辑配置文件
sudo nano /etc/frp/frpc.toml

# 2. 在文件末尾添加新的代理
[[proxies]]
name = "my-new-service"
type = "tcp"
localIP = "127.0.0.1"
localPort = 3000      # 本地服务端口
remotePort = 8005     # 公网访问端口（需服务端允许）

# 3. 重启服务
sudo systemctl restart frpc
```

### 5.3 不同场景的配置调整

#### 场景 A: 纯 SSH 远程开发
```toml
# 最小化配置
[[proxies]]
name = "ssh"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = 8000
```

#### 场景 B: Web 开发调试
```toml
# 本地 3000 端口的前端开发服务器暴露到公网
[[proxies]]
name = "dev-frontend"
type = "tcp"
localIP = "127.0.0.1"
localPort = 3000
remotePort = 8006

customDomains = ["dev.example.com"]  # 如果有域名
```

#### 场景 C: 本地 API 服务暴露
```toml
# 本地 8080 的 API 服务
[[proxies]]
name = "api-service"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8080
remotePort = 8007
```

---

## 6. 故障排除

### 6.1 连接问题

| 现象 | 可能原因 | 解决方法 |
|------|---------|---------|
| `login to server error` | 服务端未启动或防火墙 | 检查 frps 状态和 7000 端口 |
| `port not allowed` | 端口不在服务端允许范围 | 修改 frps.toml 的 allowPorts |
| `proxy name conflict` | 代理名称重复 | 确保 name 唯一 |
| `connection refused` | 本地服务未启动 | 检查本地端口监听状态 |

### 6.2 诊断命令

```bash
# 测试服务端连通性
telnet 49.234.57.210 7000

# 检查本地服务是否监听
ss -tlnp | grep :9000
ss -tlnp | grep :9001

# 查看 frpc 详细日志
sudo /usr/local/bin/frpc -c /etc/frp/frpc.toml

# 检查网络延迟
ping 49.234.57.210
```

### 6.3 日志位置

| 类型 | 位置 |
|------|------|
| Systemd 日志 | `sudo journalctl -u frpc` |
| 手动运行输出 | 终端直接输出 |
| 文件日志（需配置）| `/var/log/frpc.log` |

---

## 7. 安全建议

### 7.1 当前配置的安全注意事项

- [x] 使用 Token 认证（而非无认证）
- [ ] 建议：使用更复杂的随机 Token
- [ ] 建议：限制服务端 allowPorts 范围
- [ ] 建议：配置防火墙只允许特定 IP 访问某些端口

### 7.2 生成强 Token

```bash
# 生成随机 Token
openssl rand -base64 32

# 或使用 uuid
cat /proc/sys/kernel/random/uuid
```

---

## 8. 参考链接

- FRP 官方文档：https://gofrp.org/
- GitHub Releases：https://github.com/fatedier/frp/releases
- 配置文件详解：https://gofrp.org/zh-cn/docs/reference/configuration/

---

## 附录：完整端口速查表

### 本地端口（Ubuntu 开发机）

| 端口 | 服务 | 说明 |
|------|------|------|
| 22 | SSH | 系统自带 |
| 9000 | stream_server TCP | FIP推理输入 |
| 9001 | tcp_aggregator | ESP32数据入口 |
| 9002 | aggregator monitor | 原始TCP监控 |
| 9003 | aggregator web | WebSocket监控 |
| 8080 | stream_server HTTP | MJPEG视频流 |

### 公网端口（49.234.57.210）

| 端口 | 映射到本地 | 外部访问示例 |
|------|-----------|-------------|
| 8000 | :22 | `ssh -p 8000 user@49.234.57.210` |
| 8001 | :9001 | ESP32设备连接 |
| 8002 | :8080 | `http://49.234.57.210:8002/stream` |
| 8003 | :9002 | `nc 49.234.57.210 8003` |
| 8004 | :9003 | `http://49.234.57.210:8004/` |
| 9000 | :9000 | FIP服务直接访问 |
