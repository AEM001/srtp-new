"""Real-time FIP inference pipeline.

Receives one IMU frame at a time → runs FIP online inference → renders SMPL mesh → returns JPEG.
No intermediate files (no ODT, no CSV, no .pt files).

IMU data format per call (raw sensor coordinates, X-up / Y-left / Z-front):
    imu_data: list of 6 entries, each [ax, ay, az, roll_deg, pitch_deg, yaw_deg]
"""
import os
import sys
import numpy as np
import torch
import cv2

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

from pipeline.renderer import SMPLRenderer


_COORD_TRANSFORM = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
], dtype=np.float64)


def _euler_to_rotmat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Euler angles (degrees, ZYX / extrinsic XYZ) → 3×3 rotation matrix."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _imu_frame_to_tensors(imu_data, apply_coord_transform: bool = True):
    """Convert one IMU frame to model-ready tensors.

    Args:
        imu_data: list/array of shape [6, 6] – [[ax,ay,az,roll,pitch,yaw], ...]
        apply_coord_transform: rotate from IMU coords to model coords (default True)

    Returns:
        acc: torch.Tensor [1, 6, 3]
        ori: torch.Tensor [1, 6, 3, 3]
    """
    T = _COORD_TRANSFORM if apply_coord_transform else np.eye(3)

    acc_np = np.zeros((6, 3), dtype=np.float32)
    ori_np = np.zeros((6, 3, 3), dtype=np.float32)

    for i in range(min(len(imu_data), 6)):
        row = imu_data[i]
        ax, ay, az = float(row[0]), float(row[1]), float(row[2])
        roll, pitch, yaw = float(row[3]), float(row[4]), float(row[5])

        acc_np[i] = T @ np.array([ax, ay, az])
        R = _euler_to_rotmat(roll, pitch, yaw)
        ori_np[i] = T @ R @ T.T

    acc = torch.from_numpy(acc_np).float().unsqueeze(0)   # [1, 6, 3]
    ori = torch.from_numpy(ori_np).float().unsqueeze(0)   # [1, 6, 3, 3]
    return acc, ori


class RealtimePipeline:
    """Stateful real-time pipeline: maintains LSTM hidden states and T-pose calibration.

    Usage::

        pipeline = RealtimePipeline(model, smpl_path, device, body_params)
        # (optionally) stand in T-pose and call pipeline.calibrate()
        while streaming:
            imu_data = receive_frame()   # [[ax,ay,az,roll,pitch,yaw] × 6]
            jpeg_bytes = pipeline.process_frame(imu_data)
            send_to_viewer(jpeg_bytes)
        pipeline.cleanup()
    """

    def __init__(self, model, smpl_path: str, device, body_params: torch.Tensor,
                 width: int = 640, height: int = 480, jpeg_quality: int = 85):
        self.model = model
        self.device = device
        self.jpeg_quality = jpeg_quality

        self._body_params = body_params.unsqueeze(0).to(device)  # [1, 4]
        self.renderer = SMPLRenderer(smpl_path, device=device, width=width, height=height)

        self._integ_hc = None
        self._hip_hc = None
        self._spine_hc = None
        self._tpose_ref = None      # [15, 3, 3] – T-pose reference rotation
        self._current_raw_pose = None  # [15, 3, 3] – latest raw FIP output
        self._frame_count = 0

        self._init_model_state()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _init_model_state(self):
        """Initialise / re-initialise FIP LSTM state."""
        pose_ini = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 15, 1, 1).to(self.device)
        self.model.reset(pose_ini, self._body_params)
        self._integ_hc = None
        self._hip_hc = None
        self._spine_hc = None

    def reset(self):
        """Full reset: LSTM state + T-pose calibration + frame counter."""
        self._init_model_state()
        self._tpose_ref = None
        self._current_raw_pose = None
        self._frame_count = 0

    def calibrate(self) -> bool:
        """Store the current pose as the T-pose reference.

        Call this while the subject is in T-pose.  Returns True on success.
        """
        if self._current_raw_pose is not None:
            self._tpose_ref = self._current_raw_pose.clone()
            return True
        return False

    @property
    def is_calibrated(self) -> bool:
        return self._tpose_ref is not None

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _run_inference(self, acc: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
        """One-step FIP inference.  Returns raw pose [15, 3, 3]."""
        # Root-relative transform (mirrors inference.py logic)
        acc_root = ori[:, 5:].transpose(-1, -2) @ (acc - acc[:, 5:]).unsqueeze(-1)
        acc_root = acc_root.squeeze(-1)
        ori[:, :5] = ori[:, 5:].transpose(-1, -2) @ ori[:, :5]

        acc_in = acc_root.reshape(1, 18).to(self.device) / 30   # [1, 18]
        ori_in = ori.flatten(1).to(self.device)                  # [1, 54]

        with torch.no_grad():
            rot, _, _, _, self._integ_hc, self._spine_hc, self._hip_hc = \
                self.model.forward_online(
                    acc_in.unsqueeze(0),   # [1, 1, 18]
                    ori_in.unsqueeze(0),   # [1, 1, 54]
                    integ_hc=self._integ_hc,
                    hip_hc=self._hip_hc,
                    spine_hc=self._spine_hc,
                )

        return rot.squeeze().reshape(15, 3, 3).cpu()

    def _apply_tpose_calibration(self, raw_pose: torch.Tensor) -> torch.Tensor:
        """Apply T-pose calibration: R_cal = R_tpose^T @ R_raw."""
        if self._tpose_ref is None:
            return raw_pose
        tpose_inv = self._tpose_ref.transpose(-1, -2)                    # [15, 3, 3]
        return torch.einsum('jkl,jlm->jkm', tpose_inv, raw_pose)

    # ------------------------------------------------------------------
    # Public: process one frame
    # ------------------------------------------------------------------

    def process_frame(self, imu_data, apply_coord_transform: bool = True) -> bytes:
        """Process one IMU timestep → JPEG bytes.

        Args:
            imu_data: 6×6 array-like [[ax,ay,az,roll_deg,pitch_deg,yaw_deg], ...]
            apply_coord_transform: apply IMU→model coordinate transform (default True)

        Returns:
            JPEG-encoded rendered frame as bytes.
        """
        acc, ori = _imu_frame_to_tensors(imu_data, apply_coord_transform)

        raw_pose = self._run_inference(acc, ori)
        self._current_raw_pose = raw_pose
        self._frame_count += 1

        calibrated_pose = self._apply_tpose_calibration(raw_pose)

        vertices_list = self.renderer.poses_to_vertices(calibrated_pose.unsqueeze(0))
        frame_rgb = self.renderer.render_frame(vertices_list[0])  # [H, W, 3] uint8 RGB

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            '.jpg', frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return buf.tobytes()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        self.renderer.cleanup()
