#!/usr/bin/env python3
"""Real-time IMU → FIP inference → SMPL render → MJPEG HTTP stream.

Architecture
------------
  IMU client ──TCP:9000──> receiver threads ──queue──> processing thread
                                                              │
  Browser / VLC ──HTTP:8080/stream ──────────────────────────┘

Ports (override via env vars IMU_PORT / STREAM_PORT):
  9000  TCP  – IMU data input (JSON-lines protocol)
  8080  HTTP – MJPEG video stream + control endpoints

IMU JSON protocol (one JSON object per line, newline-terminated):
  Data frame:
    {"t": <timestamp_s>, "imus": [[ax,ay,az,roll,pitch,yaw], ...6 entries...]}
  Commands (can also be sent as HTTP GET):
    {"cmd": "calibrate"}   – set current pose as T-pose reference
    {"cmd": "reset"}       – reset LSTM state and calibration

HTTP endpoints:
  GET /           – browser UI with embedded stream
  GET /stream     – MJPEG stream (open in browser or VLC)
  GET /calibrate  – set T-pose calibration
  GET /reset      – reset pipeline
  GET /status     – JSON status (calibrated, fps, frames)
"""

import os
import sys

import os; os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

import argparse
import threading
import queue
import socket
import json
import time
import logging

import numpy as np
import cv2
from flask import Flask, Response, jsonify, render_template_string

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

from config import (
    SMPL_MODEL, MODEL_CHECKPOINT, DEVICE,
    DEFAULT_BODY_PARAMS,
    RENDER_WIDTH, RENDER_HEIGHT,
    IMU_HOST, IMU_PORT, STREAM_HOST, STREAM_PORT, STREAM_JPEG_QUALITY,
)
from pipeline.inference import load_model
from pipeline.realtime import RealtimePipeline

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Browser UI template
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>IMU 实时动作渲染</title>
<style>
  body { margin: 0; background: #111; color: #eee; font-family: sans-serif; display: flex;
         flex-direction: column; align-items: center; padding: 20px; }
  h1   { margin-bottom: 10px; font-size: 1.4rem; }
  img  { border: 2px solid #444; border-radius: 6px; max-width: 100%; }
  .bar { margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
  button { padding: 8px 18px; border-radius: 4px; border: none; cursor: pointer;
           font-size: 0.95rem; font-weight: bold; }
  .btn-cal   { background: #2ecc71; color: #000; }
  .btn-reset { background: #e74c3c; color: #fff; }
  #status { margin-top: 12px; font-size: 0.85rem; color: #aaa; }
</style>
</head>
<body>
<h1>IMU 实时动作渲染 · FIP</h1>
<img src="/stream" width="{{ width }}" height="{{ height }}" alt="stream">
<div class="bar">
  <button class="btn-cal"   onclick="cmd('calibrate')">T-pose 校准</button>
  <button class="btn-reset" onclick="cmd('reset')">重置</button>
</div>
<div id="status">等待 IMU 数据 …</div>
<script>
  async function cmd(action) {
    const r = await fetch('/' + action);
    const j = await r.json();
    document.getElementById('status').innerText = j.message || JSON.stringify(j);
  }
  async function pollStatus() {
    try {
      const r = await fetch('/status');
      const j = await r.json();
      document.getElementById('status').innerText =
        '帧数: ' + j.frames + ' | FPS: ' + j.fps.toFixed(1) +
        ' | 已校准: ' + (j.calibrated ? '✓' : '✗');
    } catch(e) {}
  }
  setInterval(pollStatus, 1000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Frame store – shared between processing thread and Flask stream generators
# ---------------------------------------------------------------------------

class FrameStore:
    """Thread-safe store for the latest rendered JPEG frame."""

    def __init__(self, placeholder: bytes):
        self._lock = threading.Condition()
        self._frame = placeholder

    def update(self, jpeg: bytes):
        with self._lock:
            self._frame = jpeg
            self._lock.notify_all()

    def get(self) -> bytes:
        with self._lock:
            return self._frame

    def wait_for_new(self, timeout: float = 1.0) -> bytes:
        """Block until a new frame is available (or timeout), then return it."""
        with self._lock:
            self._lock.wait(timeout=timeout)
            return self._frame


def _make_placeholder(width: int, height: int) -> bytes:
    """Black frame with 'Waiting for IMU...' text."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, 'Waiting for IMU data...', (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Server class
# ---------------------------------------------------------------------------

class StreamServer:
    def __init__(self, pipeline: RealtimePipeline,
                 imu_host: str, imu_port: int,
                 stream_host: str, stream_port: int):
        self.pipeline = pipeline
        self.imu_host = imu_host
        self.imu_port = imu_port
        self.stream_host = stream_host
        self.stream_port = stream_port

        self._imu_queue: queue.Queue = queue.Queue(maxsize=4)
        self._frame_store = FrameStore(_make_placeholder(RENDER_WIDTH, RENDER_HEIGHT))

        self._fps = 0.0
        self._frames_processed = 0
        self._fps_lock = threading.Lock()
        self._fps_window: list = []

        self.app = Flask(__name__)
        self._register_routes()

    # ------------------------------------------------------------------
    # Flask routes
    # ------------------------------------------------------------------

    def _register_routes(self):
        app = self.app

        @app.route('/')
        def index():
            return render_template_string(_HTML, width=RENDER_WIDTH, height=RENDER_HEIGHT)

        @app.route('/stream')
        def stream():
            return Response(
                self._mjpeg_generator(),
                mimetype='multipart/x-mixed-replace; boundary=frame',
            )

        @app.route('/calibrate')
        def calibrate():
            ok = self.pipeline.calibrate()
            msg = 'T-pose 已校准' if ok else '尚无姿态数据，请先发送 IMU 数据'
            return jsonify({'ok': ok, 'message': msg})

        @app.route('/reset')
        def reset():
            self.pipeline.reset()
            with self._fps_lock:
                self._fps_window.clear()
                self._fps = 0.0
                self._frames_processed = 0
            return jsonify({'ok': True, 'message': '已重置'})

        @app.route('/status')
        def status():
            with self._fps_lock:
                fps = self._fps
                frames = self._frames_processed
            return jsonify({
                'calibrated': self.pipeline.is_calibrated,
                'fps': round(fps, 1),
                'frames': frames,
                'imu_port': self.imu_port,
                'stream_port': self.stream_port,
            })

    def _mjpeg_generator(self):
        """Yield MJPEG frames; blocks until a new frame is ready."""
        while True:
            frame = self._frame_store.wait_for_new(timeout=1.0)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )

    # ------------------------------------------------------------------
    # Processing thread: queue → inference → render → frame store
    # ------------------------------------------------------------------

    def _processing_loop(self):
        log.info('Processing thread started.')
        self.pipeline.init_renderer()
        log.info('Renderer initialized in processing thread.')
        while True:
            try:
                msg = self._imu_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                jpeg = self.pipeline.process_frame(msg['imus'])
                self._frame_store.update(jpeg)
            except Exception as e:
                log.warning('Frame processing error: %s', e)
                continue

            elapsed = time.perf_counter() - t0
            with self._fps_lock:
                self._frames_processed += 1
                now = time.monotonic()
                self._fps_window.append(now)
                cutoff = now - 2.0
                self._fps_window = [t for t in self._fps_window if t > cutoff]
                self._fps = len(self._fps_window) / 2.0

    # ------------------------------------------------------------------
    # IMU TCP server
    # ------------------------------------------------------------------

    def _handle_client(self, conn: socket.socket, addr):
        log.info('IMU client connected: %s', addr)
        buf = b''
        try:
            while True:
                chunk = conn.recv(8192)
                if not chunk:
                    break
                buf += chunk
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode())
                    except json.JSONDecodeError as e:
                        log.debug('Bad JSON: %s – %s', line[:60], e)
                        continue

                    if 'cmd' in msg:
                        self._dispatch_cmd(msg['cmd'])
                    elif 'imus' in msg:
                        try:
                            self._imu_queue.put_nowait(msg)
                        except queue.Full:
                            pass  # drop oldest is not possible with Queue; just skip
        except OSError:
            pass
        finally:
            conn.close()
            log.info('IMU client disconnected: %s', addr)

    def _dispatch_cmd(self, cmd: str):
        if cmd == 'calibrate':
            ok = self.pipeline.calibrate()
            log.info('Calibrate command: %s', 'ok' if ok else 'no pose yet')
        elif cmd == 'reset':
            self.pipeline.reset()
            log.info('Reset command received.')
        else:
            log.warning('Unknown command: %s', cmd)

    def _imu_server_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.imu_host, self.imu_port))
        srv.listen(10)
        log.info('IMU TCP server listening on %s:%d', self.imu_host, self.imu_port)
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
            t.start()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
        proc_thread.start()

        imu_thread = threading.Thread(target=self._imu_server_loop, daemon=True)
        imu_thread.start()

        log.info('HTTP stream at http://%s:%d/', self.stream_host, self.stream_port)
        log.info('  /stream    – MJPEG video')
        log.info('  /calibrate – T-pose calibration')
        log.info('  /reset     – reset pipeline')
        log.info('  /status    – JSON status')

        self.app.run(
            host=self.stream_host,
            port=self.stream_port,
            threaded=True,
            use_reloader=False,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Real-time IMU → render → MJPEG stream server')
    p.add_argument('--imu-host',    default=IMU_HOST,    help='IMU TCP listen host (default: 0.0.0.0)')
    p.add_argument('--imu-port',    default=IMU_PORT,    type=int, help='IMU TCP listen port (default: 9000)')
    p.add_argument('--stream-host', default=STREAM_HOST, help='HTTP stream host (default: 0.0.0.0)')
    p.add_argument('--stream-port', default=STREAM_PORT, type=int, help='HTTP stream port (default: 8080)')
    p.add_argument('--width',       default=RENDER_WIDTH,  type=int)
    p.add_argument('--height',      default=RENDER_HEIGHT, type=int)
    p.add_argument('--quality',     default=STREAM_JPEG_QUALITY, type=int, help='JPEG quality 1-100')
    return p.parse_args()


def main():
    args = parse_args()

    log.info('Loading FIP model from %s …', MODEL_CHECKPOINT)
    model = load_model(MODEL_CHECKPOINT, DEVICE)
    log.info('Model loaded on %s', DEVICE)

    pipeline = RealtimePipeline(
        model=model,
        smpl_path=SMPL_MODEL,
        device=DEVICE,
        body_params=DEFAULT_BODY_PARAMS,
        width=args.width,
        height=args.height,
        jpeg_quality=args.quality,
    )

    server = StreamServer(
        pipeline=pipeline,
        imu_host=args.imu_host,
        imu_port=args.imu_port,
        stream_host=args.stream_host,
        stream_port=args.stream_port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        log.info('Shutting down …')
    finally:
        pipeline.cleanup()


if __name__ == '__main__':
    main()
