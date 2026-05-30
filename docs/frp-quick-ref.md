# FRP 快速参考卡片

## 一键重启（修改配置后）

```bash
sudo systemctl restart frpc && sudo systemctl status frpc --no-pager
```

## 端口速查

| 用途 | 本地端口 | 公网访问 |
|------|---------|---------|
| SSH | 22 | `ssh -p 8000 ubuntu@49.234.57.210` |
| ESP32入口 | 9001 | `49.234.57.210:8001` |
| 视频流 | 8080 | `http://49.234.57.210:8002/stream` |
| 监控Web | 9003 | `http://49.234.57.210:8004/` |

## 常用命令

```bash
# 启动/停止/重启
sudo systemctl start frpc
sudo systemctl stop frpc
sudo systemctl restart frpc

# 查看状态
sudo systemctl status frpc
sudo journalctl -u frpc -f

# 检查连接
pgrep -a frpc
curl http://localhost:9003/  # 测试本地web监控
```

## 新增端口步骤

1. 编辑 `sudo nano /etc/frp/frpc.toml`
2. 添加新的 `[[proxies]]` 块
3. 重启 `sudo systemctl restart frpc`

## 配置模板

```toml
[[proxies]]
name = "新服务"
type = "tcp"
localIP = "127.0.0.1"
localPort = 本地端口
remotePort = 公网端口
```

---

**完整文档**: [frp-setup-guide.md](./frp-setup-guide.md)  
**配置文件**: `/etc/frp/frpc.toml`  
**服务端**: `49.234.57.210:7000`
