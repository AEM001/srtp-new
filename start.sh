#!/bin/bash
# FIP Real-time IMU Motion Renderer - 启动脚本
# 公网配置: 49.234.57.210 (frp映射 9001->本地9001, 8080->本地8080)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="/home/albert/learn/l-vllm/.venv/bin/python"

cd "$PROJECT_DIR"

echo "======================================"
echo "FIP Real-time IMU to SMPL Renderer"
echo "公网地址: 49.234.57.210"
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
export PYOPENGL_PLATFORM=egl
echo "    渲染后端: EGL (GPU加速)"
echo "    访问地址: http://49.234.57.210:8080"
echo ""

# 后台启动 stream_server
nohup "$VENV_PYTHON" stream_server.py > /tmp/stream_server.log 2>&1 &
STREAM_PID=$!
echo "    stream_server PID: $STREAM_PID"
echo "    日志: /tmp/stream_server.log"

# 等待 stream_server 启动
sleep 5
if ! kill -0 $STREAM_PID 2>/dev/null; then
    echo "错误: stream_server 启动失败"
    tail -20 /tmp/stream_server.log
    exit 1
fi
echo "✓ stream_server 运行中"
echo ""

# 启动 tcp_aggregator
echo "[3/3] 启动 tcp_aggregator (ESP32接收端口: 9001)..."
echo "    ESP32连接目标: 49.234.57.210:9001"
echo "    需确保 frp 已将 49.234.57.210:9001 映射到本机 9001"
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
echo "  - 浏览器: http://49.234.57.210:8080"
echo "  - 状态:   http://49.234.57.210:8080/status"
echo ""
echo "ESP32配置:"
echo "  - IP: 49.234.57.210"
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
