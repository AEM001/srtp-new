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

from flask import Flask
from flask_sock import Sock

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
WEB_PORT        = 9003      # browser web monitor UI
FRAME_HZ        = 20
STALE_S         = 0.5       # drop node data older than this
SMOOTH_ALPHA    = 0.2       # low-pass filter coefficient for rpy (0=frozen, 1=raw)

NODE_NAMES = ["骨盆", "左手腕", "右手腕", "左脚踝", "右脚踝", "头部"]

_MONITOR_HTML = """\
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IMU 实时监控</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #c9d1d9; font-family: 'Consolas', 'Menlo', monospace;
       padding: 16px; }
h1 { font-size: 1.1rem; color: #58a6ff; margin-bottom: 4px; }
#meta { font-size: 0.78rem; color: #8b949e; margin-bottom: 12px; }
#warn { color: #f0a500; font-size: 0.82rem; min-height: 1.2em; margin-bottom: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th { background: #161b22; color: #58a6ff; padding: 6px 8px; text-align: right;
     border-bottom: 1px solid #30363d; font-weight: normal; }
th:first-child { text-align: left; }
td { padding: 5px 8px; text-align: right; border-bottom: 1px solid #21262d; }
td:first-child { text-align: left; color: #e6edf3; }
.ok   { color: #3fb950; }
.warn { color: #f0a500; }
.bad  { color: #f85149; }
.lag  { color: #a371f7; }
.nodata { color: #484f58; font-style: italic; }
#dot { display:inline-block; width:8px; height:8px; border-radius:50%;
       background:#3fb950; margin-right:6px; }
#dot.off { background:#f85149; }
</style>
</head>
<body>
<h1><span id="dot" class="off"></span>IMU 实时稳定性监控</h1>
<div id="meta">等待连接...</div>
<div id="warn"></div>
<table>
  <thead>
    <tr>
      <th>节点</th>
      <th>roll</th><th>pitch</th><th>yaw</th>
      <th>std_r</th><th>std_p</th>
      <th>跳变_r</th><th>跳变_p</th>
      <th>延迟</th><th>状态</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>
<script>
const NAMES = ["骨盆","左手腕","右手腕","左脚踝","右脚踝","头部"];
const HIST  = 30;
const hist  = Array.from({length:6}, ()=>[]);
let frameCount = 0, t0 = Date.now(), ws;

function fmt(v, d=1){ return v==null?"--":v.toFixed(d); }
function cls(v, warn, bad){
  if(v==null) return "";
  if(v>=bad)  return "bad";
  if(v>=warn) return "warn";
  return "ok";
}

function std(arr){ // arr: number[]
  if(arr.length<2) return 0;
  const m = arr.reduce((a,b)=>a+b,0)/arr.length;
  return Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/arr.length);
}

function render(data){
  const imus    = data.imus;
  const ages    = data.ages_ms || Array(6).fill(null);
  frameCount++;
  const fps     = (frameCount/((Date.now()-t0)/1000)).toFixed(1);

  // update history
  imus.forEach((imu, i)=>{
    const nonzero = imu.some(v=>v!==0);
    if(nonzero){
      hist[i].push(imu);
      if(hist[i].length>HIST) hist[i].shift();
    }
  });

  // timestamp skew
  const validAges = ages.filter(a=>a!=null);
  const skew = validAges.length>1 ? Math.max(...validAges)-Math.min(...validAges) : 0;
  document.getElementById('warn').textContent =
    skew > 100 ? `⚠ 节点间时间戳偏差 ${skew}ms（>100ms 可能影响同步）` : '';

  document.getElementById('dot').className = 'on' ;
  document.getElementById('meta').textContent =
    `帧: ${frameCount}  FPS: ${fps}  节点: ${imus.filter((_,i)=>ages[i]!=null).length}/6`;

  const rows = NAMES.map((name,i)=>{
    const h = hist[i];
    const age = ages[i];
    const ageStr = age==null ? '--' : age+'ms';
    const ageCls = age!=null && age>100 ? 'lag' : 'ok';

    if(h.length<2){
      return `<tr><td>${name}</td><td colspan="8" class="nodata">无数据</td></tr>`;
    }
    const rolls  = h.map(r=>r[3]);
    const pitchs = h.map(r=>r[4]);
    const latest = h[h.length-1];
    const prev   = h[h.length-2];

    const roll=latest[3], pitch=latest[4], yaw=latest[5];
    const sr=std(rolls), sp=std(pitchs);
    const dr=Math.abs(latest[3]-prev[3]), dp=Math.abs(latest[4]-prev[4]);

    const stdBad  = sr>3||sp>3;
    const jumpBad = dr>5||dp>5;
    let flag, flagCls;
    if(stdBad&&jumpBad){ flag='⚠ 抖动+跳变'; flagCls='bad'; }
    else if(stdBad)    { flag='⚠ 抖动';      flagCls='warn'; }
    else if(jumpBad)   { flag='⡱ 跳变';      flagCls='warn'; }
    else               { flag='✓ 稳定';       flagCls='ok'; }
    if(age!=null&&age>100) flag+=' ⏱';

    return `<tr>
      <td>${name}</td>
      <td>${fmt(roll)}</td><td>${fmt(pitch)}</td><td>${fmt(yaw)}</td>
      <td class="${cls(sr,1.5,3)}">${fmt(sr,2)}</td>
      <td class="${cls(sp,1.5,3)}">${fmt(sp,2)}</td>
      <td class="${cls(dr,2.5,5)}">${fmt(dr,2)}</td>
      <td class="${cls(dp,2.5,5)}">${fmt(dp,2)}</td>
      <td class="${ageCls}">${ageStr}</td>
      <td class="${flagCls}">${flag}</td>
    </tr>`;
  });
  document.getElementById('tbody').innerHTML = rows.join('');
}

function connect(){
  const proto = location.protocol==='https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto+'://'+location.host+'/ws');
  ws.onopen  = ()=>{ document.getElementById('dot').className='on'; };
  ws.onclose = ()=>{ document.getElementById('dot').className='off';
                     setTimeout(connect, 2000); };
  ws.onerror = ()=>ws.close();
  ws.onmessage = e=>{ try{ render(JSON.parse(e.data)); }catch(ex){} };
}
connect();
</script>
</body>
</html>
"""


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

        # monitor broadcast clients (raw TCP)
        self._monitor_clients: list[socket.socket] = []
        self._monitor_lock = threading.Lock()

        # websocket broadcast clients (web UI)
        self._ws_clients: list = []
        self._ws_lock = threading.Lock()

        # latest assembled monitor frame for new WS connections
        self._latest_monitor_frame: str | None = None

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
        self._latest_monitor_frame = line
        # TCP clients
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
        # WebSocket clients
        with self._ws_lock:
            dead = []
            for ws in self._ws_clients:
                try:
                    ws.send(line)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._ws_clients.remove(ws)

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
            ages = [None] * 6  # data age per node (seconds), None = no data
            fresh_count = 0

            with self._lock:
                if not self._nodes:
                    log.debug("No nodes connected")
                    continue
                for idx, d in self._nodes.items():
                    age = now - d["t"]
                    ages[idx] = round(age * 1000)  # ms
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
                # broadcast includes ages for monitor display
                monitor_frame = {"t": now, "imus": imus, "ages_ms": ages}
                self._broadcast_monitor(json.dumps(monitor_frame))

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

    def _web_monitor_loop(self):
        """Flask + WebSocket web monitor on WEB_PORT."""
        app  = Flask(__name__)
        sock = Sock(app)

        agg = self  # closure

        @app.route("/")
        def index():
            return _MONITOR_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

        @sock.route("/ws")
        def ws_handler(ws):
            with agg._ws_lock:
                agg._ws_clients.append(ws)
            # send latest frame immediately so the page isn't blank
            if agg._latest_monitor_frame:
                try:
                    ws.send(agg._latest_monitor_frame)
                except Exception:
                    pass
            try:
                while True:
                    ws.receive(timeout=30)  # keep alive; we only push, never read
            except Exception:
                pass
            finally:
                with agg._ws_lock:
                    if ws in agg._ws_clients:
                        agg._ws_clients.remove(ws)

        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.WARNING)
        log.info("Web monitor on http://0.0.0.0:%d/", WEB_PORT)
        app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)

    def run(self):
        self._connect_stream()

        threading.Thread(target=self._listen_loop, daemon=True).start()
        threading.Thread(target=self._forward_loop, daemon=True).start()
        threading.Thread(target=self._monitor_listen_loop, daemon=True).start()
        threading.Thread(target=self._web_monitor_loop, daemon=True).start()

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
