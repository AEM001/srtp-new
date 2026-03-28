"""SMPL mesh rendering via pyrender (headless, Linux).

Converts calibrated pose .pt files → SMPL vertices → rendered MP4 video.
Designed for reuse: SMPLRenderer can render single frames (for real-time)
or full sequences (for offline batch rendering).

Set PYOPENGL_PLATFORM env var before import:
  osmesa  – software rendering, works in Docker without GPU (default)
  egl     – hardware-accelerated, requires GPU + EGL drivers
"""
import os

if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import sys
import numpy as np
import torch
import trimesh
import pyrender
import imageio
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

from src.kinematic_model import ParametricModel
from src.eval_tools import glb2local


class SMPLRenderer:
    """Encapsulates SMPL model + pyrender scene for efficient rendering."""

    def __init__(self, smpl_path, device='cpu', width=640, height=480):
        self.device = device
        self.width = width
        self.height = height

        # SMPL body model
        self.body_model = ParametricModel(smpl_path, device=device)
        self.faces = self.body_model.face

        # Pyrender scene (reused across frames)
        self._build_scene()

    def _build_scene(self):
        """Create pyrender scene with camera + lights."""
        self.scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0],
            ambient_light=[0.3, 0.3, 0.3],
        )

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 2.5
        self.scene.add(camera, pose=cam_pose)

        self.scene.add(
            pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0),
            pose=cam_pose,
        )
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = [0, 1, 2]
        self.scene.add(
            pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0),
            pose=fill_pose,
        )

        self._offscreen = pyrender.OffscreenRenderer(self.width, self.height)
        self._mesh_node = None

    # ---------- Core methods ----------

    def poses_to_vertices(self, poses):
        """Convert [T, 15, 3, 3] global poses → list of [V, 3] vertex arrays."""
        poses_local = glb2local(poses)
        vertices = []
        with torch.no_grad():
            for i in range(len(poses_local)):
                p = poses_local[i:i+1].to(self.device)
                p[:, [0, 7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=self.device)
                _, _, mesh = self.body_model.forward_kinematics(p, calc_mesh=True)
                vertices.append(mesh[0].cpu().numpy())
        return vertices

    def render_frame(self, vertices):
        """Render a single set of vertices → RGB uint8 array [H, W, 3]."""
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces)
        mesh.visual.vertex_colors = [100, 150, 200, 255]
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        if self._mesh_node is not None:
            self.scene.remove_node(self._mesh_node)
        self._mesh_node = self.scene.add(py_mesh)

        color, _ = self._offscreen.render(self.scene)
        return color

    def render_video(self, pose_path, output_path, fps=30):
        """Render a calibrated .pt pose file to an MP4 video.

        Returns the number of rendered frames.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        data = torch.load(pose_path, map_location=self.device, weights_only=False)
        poses = data['poses']

        vertices_list = self.poses_to_vertices(poses)

        writer = imageio.get_writer(output_path, fps=fps, format='FFMPEG', codec='libx264')
        for verts in tqdm(vertices_list, desc=f"    Rendering"):
            frame = self.render_frame(verts)
            writer.append_data(frame)
        writer.close()

        return len(vertices_list)

    def cleanup(self):
        """Release GPU / offscreen renderer resources."""
        self._offscreen.delete()
