#!/bin/bash
# FIP Real-time IMU Motion Renderer - Mac 本地启动脚本
# ESP32 通过本地 Wi-Fi 直连到本机局域网 IP:9001

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

cd "$PROJECT_DIR"

# 获取本机局域网 IP
LAN_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
if [ -z "$LAN_IP" ]; then
    LAN_IP="<本机局域网 IP>"
fi

echo "======================================"
echo "FIP Real-time IMU to SMPL Renderer"
echo "======================================"
echo ""

# 检查venv
echo "[1/3] 检查 Python 环境..."
if [ ! -f "$VENV_PYTHON" ]; then
    echo "错误: 未找到 venv Python: $VENV_PYTHON"
    exit 1
fi
echo "✓ Python: $VENV_PYTHON"
echo ""

# 启动 stream_server
echo "[2/3] 启动 stream_server (HTTP:8080, TCP:9000)..."
echo "    渲染后端: VTK (macOS 原生)"
echo "    浏览器访问: http://localhost:8080"
echo ""

nohup "$VENV_PYTHON" stream_server.py > /tmp/stream_server.log 2>&1 &
STREAM_PID=$!
echo "    stream_server PID: $STREAM_PID"
echo "    日志: /tmp/stream_server.log"

sleep 5
if ! kill -0 $STREAM_PID 2>/dev/null; then
    echo "错误: stream_server 启动失败"
    tail -20 /tmp/stream_server.log
    exit 1
fi
echo "✓ stream_server 运行中"
echo ""

# 启动 tcp_aggregator
echo "[3/3] 启动 tcp_aggregator (ESP32 接收端口: 9001)..."
echo "    ESP32 连接目标: $LAN_IP:9001"
echo ""

nohup "$VENV_PYTHON" tcp_aggregator.py > /tmp/tcp_aggregator.log 2>&1 &
AGG_PID=$!
echo "    tcp_aggregator PID: $AGG_PID"
echo "    日志: /tmp/tcp_aggregator.log"

sleep 2
if ! kill -0 $AGG_PID 2>/dev/null; then
    echo "错误: tcp_aggregator 启动失败"
    tail -20 /tmp/tcp_aggregator.log
    exit 1
fi
echo "✓ tcp_aggregator 运行中"
echo ""

echo "======================================"
echo "所有服务已启动!"
echo ""
echo "访问地址:"
echo "  - 浏览器: http://localhost:8080"
echo "  - 状态:   http://localhost:8080/status"
echo ""
echo "ESP32 配置:"
echo "  - IP: $LAN_IP (Mac 局域网 IP)"
echo "  - 端口: 9001"
echo "  - NODE_INDEX: 0=骨盆, 1=左手腕, 2=右手腕, 3=左脚踝, 4=右脚踝, 5=头部"
echo ""
echo "日志文件:"
echo "  - stream_server:     /tmp/stream_server.log"
echo "  - tcp_aggregator:    /tmp/tcp_aggregator.log"
echo ""
echo "停止命令:"
echo "  kill $STREAM_PID $AGG_PID"
echo "======================================"

# 保存PID供停止用
echo "$STREAM_PID $AGG_PID" > /tmp/fip_pids.txt
