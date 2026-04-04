#!/usr/bin/env python3
"""TCP aggregator: receives per-node IMU frames from ESP32 boards,
assembles them into a 6-IMU frame, and forwards to stream_server.py.

Each ESP32 connects independently and sends one line per timestep:
  {"node": 3, "t": 12.3, "acc": [ax,ay,az], "rpy": [roll,pitch,yaw]}\n

This program:
  1. Listens on AGGREGATOR_PORT (default 9001) for ESP32 connections
  2. Assembles the latest reading from each node into a 6-IMU frame
  3. Forwards frames at FRAME_HZ to stream_server TCP port (default 9000)

Usage:
  python tcp_aggregator.py
  python tcp_aggregator.py --listen-port 9001 --stream-host 127.0.0.1 --stream-port 9000
"""

import argparse
import json
import logging
import socket
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AGGREGATOR_HOST = "0.0.0.0"  # 监听所有接口，接收公网ESP32连接
AGGREGATOR_PORT = 9001      # ESP32 boards connect HERE (需frp映射到49.234.57.210:9001)
STREAM_HOST     = "127.0.0.1"
STREAM_PORT     = 9000      # stream_server.py listens here
MONITOR_PORT    = 9002      # monitor clients connect here to receive assembled frames
FRAME_HZ        = 20
STALE_S         = 0.5       # drop node data older than this
SMOOTH_ALPHA    = 0.2       # low-pass filter coefficient for rpy (0=frozen, 1=raw)


class Aggregator:
    def __init__(self, listen_host, listen_port, stream_host, stream_port):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.stream_host = stream_host
        self.stream_port = stream_port

        # latest data per node index 0-5
        self._nodes: dict[int, dict] = {}
        self._lock = threading.Lock()

        # smoothed rpy per node for low-pass filter
        self._smooth_rpy: dict[int, list] = {}

        self._stream_sock: socket.socket | None = None
        self._stream_lock = threading.Lock()

        # monitor broadcast clients
        self._monitor_clients: list[socket.socket] = []
        self._monitor_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Stream server TCP connection
    # ------------------------------------------------------------------

    def _connect_stream(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.stream_host, self.stream_port))
                with self._stream_lock:
                    self._stream_sock = s
                log.info("Connected to stream_server %s:%d", self.stream_host, self.stream_port)
                return
            except OSError as e:
                log.warning("stream_server connect failed (%s), retry in 3s…", e)
                time.sleep(3)

    def _send_stream(self, line: str):
        with self._stream_lock:
            sock = self._stream_sock
        if not sock:
            return
        try:
            sock.sendall((line + "\n").encode())
        except OSError as e:
            log.warning("stream_server send error: %s – reconnecting…", e)
            with self._stream_lock:
                self._stream_sock = None
            threading.Thread(target=self._connect_stream, daemon=True).start()

    def _broadcast_monitor(self, line: str):
        data = (line + "\n").encode()
        with self._monitor_lock:
            dead = []
            for s in self._monitor_clients:
                try:
                    s.sendall(data)
                except OSError:
                    dead.append(s)
            for s in dead:
                self._monitor_clients.remove(s)

    def _monitor_listen_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", MONITOR_PORT))
        srv.listen(5)
        log.info("Monitor broadcast listening on 127.0.0.1:%d", MONITOR_PORT)
        while True:
            conn, addr = srv.accept()
            log.info("Monitor client connected: %s", addr)
            with self._monitor_lock:
                self._monitor_clients.append(conn)

    # ------------------------------------------------------------------
    # Handle one ESP32 client
    # ------------------------------------------------------------------

    def _handle_node(self, conn: socket.socket, addr):
        log.info("Node connected: %s", addr)
        buf = b""
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode())
                    except json.JSONDecodeError:
                        continue

                    node_idx = int(msg.get("node", -1))
                    if not 0 <= node_idx <= 5:
                        continue

                    acc = msg.get("acc", [0, 0, 0])
                    rpy = msg.get("rpy", [0, 0, 0])

                    # yaw is gyro-only (no correction) → drifts unboundedly, zero it out
                    rpy[2] = 0.0

                    with self._lock:
                        if node_idx not in self._smooth_rpy:
                            self._smooth_rpy[node_idx] = list(rpy)
                        else:
                            s = self._smooth_rpy[node_idx]
                            for k in range(3):
                                s[k] = SMOOTH_ALPHA * rpy[k] + (1 - SMOOTH_ALPHA) * s[k]

                        self._nodes[node_idx] = {
                            "acc": acc,
                            "rpy": list(self._smooth_rpy[node_idx]),
                            "t":   time.monotonic(),
                        }
        except OSError:
            pass
        finally:
            conn.close()
            log.info("Node disconnected: %s", addr)

    # ------------------------------------------------------------------
    # Frame assembly loop
    # ------------------------------------------------------------------

    def _forward_loop(self):
        interval = 1.0 / FRAME_HZ
        log.info("Forwarding at %d Hz to stream_server", FRAME_HZ)
        next_tick = time.monotonic()
        frame_count = 0

        while True:
            next_tick += interval
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

            if not self._stream_sock:
                log.debug("No stream socket, skipping")
                continue

            now = time.monotonic()
            imus = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(6)]
            fresh_count = 0

            with self._lock:
                if not self._nodes:
                    log.debug("No nodes connected")
                    continue
                for idx, d in self._nodes.items():
                    age = now - d["t"]
                    if age > STALE_S:
                        log.debug("Node %d data stale (%.2fs old)", idx, age)
                        continue
                    fresh_count += 1
                    ax, ay, az = d["acc"]
                    ro, pi, ya = d["rpy"]
                    imus[idx] = [ax, ay, az, ro, pi, ya]

            if fresh_count > 0:
                frame_count += 1
                if frame_count <= 5 or frame_count % 50 == 0:
                    log.info("Sending frame #%d with %d fresh nodes", frame_count, fresh_count)
                frame = {"t": now, "imus": imus}
                serialized = json.dumps(frame)
                self._send_stream(serialized)
                self._broadcast_monitor(serialized)

    # ------------------------------------------------------------------
    # Listen for ESP32 connections
    # ------------------------------------------------------------------

    def _listen_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.listen_host, self.listen_port))
        srv.listen(10)
        log.info("Listening for ESP32 nodes on %s:%d", self.listen_host, self.listen_port)
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=self._handle_node, args=(conn, addr), daemon=True).start()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        self._connect_stream()

        threading.Thread(target=self._listen_loop, daemon=True).start()
        threading.Thread(target=self._forward_loop, daemon=True).start()
        threading.Thread(target=self._monitor_listen_loop, daemon=True).start()

        log.info("Aggregator running. ESP32 → port %d  |  stream_server ← port %d",
                 self.listen_port, self.stream_port)
        try:
            while True:
                time.sleep(5)
                with self._lock:
                    active = sorted(self._nodes.keys())
                log.info("Active nodes: %s", active)
        except KeyboardInterrupt:
            log.info("Stopped.")


def main():
    p = argparse.ArgumentParser(description="TCP aggregator: ESP32 nodes → stream_server")
    p.add_argument("--listen-host",  default=AGGREGATOR_HOST)
    p.add_argument("--listen-port",  default=AGGREGATOR_PORT, type=int,
                   help="Port ESP32 boards connect to (default: 9001)")
    p.add_argument("--stream-host",  default=STREAM_HOST)
    p.add_argument("--stream-port",  default=STREAM_PORT, type=int,
                   help="stream_server port (default: 9000)")
    args = p.parse_args()

    Aggregator(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        stream_host=args.stream_host,
        stream_port=args.stream_port,
    ).run()


if __name__ == "__main__":
    main()
