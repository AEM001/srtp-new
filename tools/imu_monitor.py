#!/usr/bin/env python3
"""实时IMU数据稳定性监控

连接到 tcp_aggregator 的 monitor 广播端口 (9002)，实时显示：
  - 每个节点的当前 acc/rpy 值
  - 最近30帧的标准差（衡量抖动）
  - 帧间跳变量（相邻帧差值）

Usage:
    python tools/imu_monitor.py
"""

import socket
import json
import time
import sys
import os
from collections import deque
import numpy as np

HOST = "127.0.0.1"
PORT = 9002  # aggregator monitor broadcast port

HISTORY = 30  # 用于计算抖动的历史帧数
NODE_NAMES = ["骨盆  ", "左手腕", "右手腕", "左脚踝", "右脚踝", "头部  "]


def clear():
    os.system("clear")


def std_bar(val, max_val=10.0, width=15):
    filled = int(min(val / max_val, 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def main():
    print(f"连接到 {HOST}:{PORT} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
    except ConnectionRefusedError:
        print("连接失败！请确认 tcp_aggregator.py 正在运行（需要重启以加载monitor端口）。")
        sys.exit(1)

    sock.settimeout(5.0)
    print("已连接，等待IMU数据...\n")

    buf = b""
    history = [deque(maxlen=HISTORY) for _ in range(6)]
    prev_frame = None
    frame_count = 0
    t_start = time.monotonic()
    last_recv = time.monotonic()

    try:
        while True:
            try:
                chunk = sock.recv(8192)
                if not chunk:
                    print("连接断开")
                    break
                buf += chunk
                last_recv = time.monotonic()
            except socket.timeout:
                elapsed = time.monotonic() - last_recv
                print(f"\r⚠ 等待数据... ({elapsed:.1f}s)", end="", flush=True)
                continue

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue
                if "imus" not in msg:
                    continue

                imus = msg["imus"]
                ages_ms = msg.get("ages_ms", [None] * 6)
                frame_count += 1

                for i, imu in enumerate(imus[:6]):
                    if len(imu) == 6 and any(v != 0.0 for v in imu):
                        history[i].append(imu)

                elapsed = time.monotonic() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0.0

                # 计算各节点时间戳偏差（相对于最小age的节点）
                valid_ages = [a for a in ages_ms if a is not None]
                min_age = min(valid_ages) if valid_ages else 0
                max_age = max(valid_ages) if valid_ages else 0
                skew_warn = (max_age - min_age) > 100  # >100ms 节点间时差

                clear()
                print(f"══ IMU 实时稳定性监控  帧:{frame_count}  FPS:{fps:.1f}  Ctrl+C退出 ══")
                if skew_warn:
                    print(f"  ⚠ 节点间时间戳偏差 {max_age - min_age}ms（>100ms 可能影响同步）")
                print()
                print(f"{'节点':<8}  {'roll':>7} {'pitch':>7} {'yaw':>7}  "
                      f"{'std_r':>6} {'std_p':>6} {'std_y':>6}  "
                      f"{'跳变_r':>7} {'跳变_p':>7} {'跳变_y':>7}  {'延迟':>6}  状态")
                print("─" * 108)

                for i in range(6):
                    h = history[i]
                    age_str = f"{ages_ms[i]}ms" if ages_ms[i] is not None else "  --"
                    if len(h) < 2:
                        print(f"{NODE_NAMES[i]}  {'(无数据)':<62}  {age_str:>6}")
                        continue

                    arr = np.array(h)       # [N, 6]
                    latest = arr[-1]
                    std = arr.std(axis=0)   # per-channel std

                    roll, pitch, yaw = latest[3], latest[4], latest[5]
                    sr, sp, sy = std[3], std[4], std[5]

                    # 帧间跳变：相邻两帧的差值
                    delta = np.abs(arr[-1] - arr[-2])
                    dr, dp, dy = delta[3], delta[4], delta[5]

                    # 判断不稳定
                    std_bad  = sr > 3.0 or sp > 3.0 or sy > 3.0
                    jump_bad = dr > 5.0 or dp > 5.0 or dy > 5.0
                    age_bad  = ages_ms[i] is not None and ages_ms[i] > 100
                    if std_bad and jump_bad:
                        flag = "⚠ 抖动+跳变"
                    elif std_bad:
                        flag = "⚠ 抖动"
                    elif jump_bad:
                        flag = "⡱ 跳变"
                    else:
                        flag = "✓ 稳定"
                    if age_bad:
                        flag += " ⏱滞后"

                    print(
                        f"{NODE_NAMES[i]}  "
                        f"{roll:>7.1f} {pitch:>7.1f} {yaw:>7.1f}  "
                        f"{sr:>6.2f} {sp:>6.2f} {sy:>6.2f}  "
                        f"{dr:>7.2f} {dp:>7.2f} {dy:>7.2f}  {age_str:>6}  {flag}"
                    )

                print()
                print(f"  std 抖动阈值 >3° 为不稳定 | 跳变阈值 >5° 为异常跳帧 | 延迟 >100ms 为节点滞后")
                prev_frame = imus

    except KeyboardInterrupt:
        print("\n\n已停止监控。")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
