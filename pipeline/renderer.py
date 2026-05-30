"""SMPL mesh rendering via VTK (headless, cross-platform).

Converts calibrated pose .pt files → SMPL vertices → rendered MP4 video.
Replaces pyrender/EGL/OSMesa with VTK off-screen rendering for macOS compatibility.
"""
import os
import sys
import numpy as np
import torch
import imageio
from tqdm import tqdm

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

from src.kinematic_model import ParametricModel
from src.eval_tools import glb2local


class SMPLRenderer:
    """Encapsulates SMPL model + VTK off-screen renderer."""

    def __init__(self, smpl_path, device='cpu', width=640, height=480):
        self.device = device
        self.width = width
        self.height = height

        # SMPL body model
        self.body_model = ParametricModel(smpl_path, device=device)
        self.faces = self.body_model.face

        # VTK off-screen render pipeline
        self._build_scene()

    def _build_scene(self):
        """Create VTK renderer, window, camera and lights."""
        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_renderer.SetBackground(1.0, 1.0, 1.0)

        # Camera: z=2.5, look at origin, yfov ~60° (pi/3)
        camera = vtk.vtkCamera()
        camera.SetPosition(0, 0, 2.5)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
        camera.SetViewAngle(60.0)
        self.vtk_renderer.SetActiveCamera(camera)

        # Lights
        light_dir = vtk.vtkLight()
        light_dir.SetLightTypeToCameraLight()
        light_dir.SetIntensity(1.0)
        light_dir.SetColor(1.0, 1.0, 1.0)
        self.vtk_renderer.AddLight(light_dir)

        light_pt = vtk.vtkLight()
        light_pt.SetLightTypeToSceneLight()
        light_pt.SetPosition(0, 1, 2)
        light_pt.SetIntensity(0.5)
        light_pt.SetColor(1.0, 1.0, 1.0)
        self.vtk_renderer.AddLight(light_pt)

        # Off-screen window
        self.vtk_renWin = vtk.vtkRenderWindow()
        self.vtk_renWin.SetOffScreenRendering(1)
        self.vtk_renWin.SetSize(self.width, self.height)
        self.vtk_renWin.AddRenderer(self.vtk_renderer)

        # Mesh geometry (reused; only point positions change each frame)
        self.vtk_points = vtk.vtkPoints()
        self.vtk_cells = vtk.vtkCellArray()
        self.vtk_polydata = vtk.vtkPolyData()
        self.vtk_polydata.SetPoints(self.vtk_points)
        self.vtk_polydata.SetPolys(self.vtk_cells)

        self.vtk_mapper = vtk.vtkPolyDataMapper()
        self.vtk_mapper.SetInputData(self.vtk_polydata)

        self.vtk_actor = vtk.vtkActor()
        self.vtk_actor.SetMapper(self.vtk_mapper)
        self.vtk_actor.GetProperty().SetColor(100 / 255, 150 / 255, 200 / 255)
        self.vtk_renderer.AddActor(self.vtk_actor)

        # Pre-build face connectivity (faces don't change)
        for f in self.faces:
            self.vtk_cells.InsertNextCell(3, [int(f[0]), int(f[1]), int(f[2])])

        # Reusable window-to-image filter
        self._w2i = vtk.vtkWindowToImageFilter()
        self._w2i.SetInput(self.vtk_renWin)
        self._w2i.SetInputBufferTypeToRGB()

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
        n = vertices.shape[0]
        if self.vtk_points.GetNumberOfPoints() != n:
            self.vtk_points.SetNumberOfPoints(n)
        pts = numpy_to_vtk(vertices, deep=True)
        self.vtk_points.SetData(pts)
        self.vtk_points.Modified()
        self.vtk_polydata.Modified()

        self.vtk_renWin.Render()
        self._w2i.Modified()
        self._w2i.Update()

        img = self._w2i.GetOutput()
        arr = vtk_to_numpy(img.GetPointData().GetScalars())
        arr = arr.reshape(self.height, self.width, 3)
        # VTK origin is bottom-left; flip vertically for normal image orientation
        arr = np.flipud(arr)
        return arr

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
        """Release VTK resources."""
        self.vtk_renWin.Finalize()
