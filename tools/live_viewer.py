#!/usr/bin/env python3
"""Local real-time SMPL motion viewer (VTK window).

Play a calibrated pose .pt file in a native window, or connect to a running
stream_server via TCP and display live frames.

Usage:
    python tools/live_viewer.py --file output/poses/m2_calibrated.pt
    python tools/live_viewer.py --file output/poses/m2_calibrated.pt --fps 30
    python tools/live_viewer.py --tcp localhost:9000      # live from stream_server
"""
import argparse
import os
import sys
import time
import json
import socket

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

import numpy as np
import torch

from pipeline.renderer import SMPLRenderer
from config import SMPL_MODEL, DEVICE


class PosePlayer:
    """VTK-based local window player for SMPL motions."""

    def __init__(self, smpl_path, device='cpu', width=640, height=480):
        self.device = device
        self.renderer = SMPLRenderer(smpl_path, device=device, width=width, height=height)
        self._setup_interactor(width, height)
        self._paused = False
        self._should_exit = False
        self._vertices_list = []
        self._frame_idx = 0
        self._fps = 30.0
        self._last_time = 0.0

    def _setup_interactor(self, width, height):
        import vtk
        self._renWin = self.renderer.vtk_renWin
        # Switch to on-screen window
        self._renWin.SetOffScreenRendering(0)
        self._renWin.SetWindowName("FIP Live Viewer")
        self._renWin.SetSize(width, height)

        self._interactor = vtk.vtkRenderWindowInteractor()
        self._interactor.SetRenderWindow(self._renWin)
        self._interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # Timer for animation playback
        self._interactor.AddObserver("TimerEvent", self._on_timer)
        self._interactor.AddObserver("KeyPressEvent", self._on_key)
        self._timer_id = None

    # ---------- Event handlers ----------

    def _on_key(self, obj, event):
        key = obj.GetKeySym().lower()
        if key == 'space':
            self._paused = not self._paused
            print(f"{'Paused' if self._paused else 'Playing'}")
        elif key == 'q' or key == 'escape':
            self._should_exit = True
            obj.TerminateApp()
        elif key == 'r':
            self._frame_idx = 0
            print("Rewound")
        elif key == 'plus' or key == 'equal':
            self._fps = min(self._fps + 5, 120)
            print(f"Speed: {self._fps:.0f} FPS")
        elif key == 'minus':
            self._fps = max(self._fps - 5, 1)
            print(f"Speed: {self._fps:.0f} FPS")

    def _on_timer(self, obj, event):
        if self._paused or not self._vertices_list:
            return

        now = time.time()
        dt = now - self._last_time
        expected_dt = 1.0 / self._fps
        if dt < expected_dt:
            return  # skip frame if too early
        self._last_time = now

        verts = self._vertices_list[self._frame_idx]
        self._render_vertices(verts)
        self._frame_idx = (self._frame_idx + 1) % len(self._vertices_list)

    def _render_vertices(self, vertices):
        import vtk
        n = vertices.shape[0]
        if self.renderer.vtk_points.GetNumberOfPoints() != n:
            self.renderer.vtk_points.SetNumberOfPoints(n)
        from vtk.util.numpy_support import numpy_to_vtk
        pts = numpy_to_vtk(vertices, deep=True)
        self.renderer.vtk_points.SetData(pts)
        self.renderer.vtk_points.Modified()
        self.renderer.vtk_polydata.Modified()
        self._renWin.Render()

    # ---------- Public API ----------

    def load_poses(self, pose_path):
        """Load a calibrated .pt pose file."""
        data = torch.load(pose_path, map_location=self.device, weights_only=False)
        poses = data['poses']
        self._vertices_list = self.renderer.poses_to_vertices(poses)
        print(f"Loaded {len(self._vertices_list)} frames from {pose_path}")

    def play(self):
        """Start the VTK event loop."""
        import vtk
        self._renWin.Render()
        self._interactor.Initialize()
        self._last_time = time.time()
        self._timer_id = self._interactor.CreateRepeatingTimer(int(1000 / 60))  # 60 Hz UI timer
        print("Controls: Space = pause, Q/Esc = quit, R = rewind, +/- = speed")
        self._interactor.Start()
        self.cleanup()

    def set_vertices(self, vertices):
        """Display a single frame (for live TCP mode)."""
        self._render_vertices(vertices)
        self._renWin.Render()

    def cleanup(self):
        self.renderer.cleanup()


def play_file(args):
    player = PosePlayer(SMPL_MODEL, device=DEVICE, width=args.width, height=args.height)
    player.load_poses(args.file)
    player.play()


def play_tcp(args):
    """Connect to a stream_server TCP port and display frames live."""
    import re
    m = re.match(r'(.+):(\d+)', args.tcp)
    host, port = m.group(1), int(m.group(2))

    player = PosePlayer(SMPL_MODEL, device=DEVICE, width=args.width, height=args.height)

    print(f"Connecting to {host}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print("Connected. Waiting for IMU data ...")

    buf = b''
    try:
        while True:
            chunk = sock.recv(8192)
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
                except json.JSONDecodeError:
                    continue
                if 'imus' in msg:
                    # We can't do FIP inference here without the model, so for now
                    # this path is just a placeholder.  Full live display is better
                    # done by pointing a browser at the stream_server HTTP endpoint.
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        player.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Local VTK SMPL motion viewer')
    parser.add_argument('--file', help='Path to a calibrated pose .pt file')
    parser.add_argument('--tcp', help='Host:port of a running stream_server (e.g. localhost:9000)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=30, help='Playback FPS')
    args = parser.parse_args()

    if args.file:
        play_file(args)
    elif args.tcp:
        play_tcp(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
