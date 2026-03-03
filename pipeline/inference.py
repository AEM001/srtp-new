"""FIP model inference and T-pose calibration.

Merges the old infer_new_data.py + apply_tpose_calibration.py into one step.
Pipeline: processed CSV → FIP online inference → T-pose calibration → calibrated poses.pt
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

from model.net import FIP


# ---------- Helpers ----------

def _euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """Euler angles (degrees, ZYX convention) → 3x3 rotation matrix."""
    r, p, y = np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def csv_to_tensors(csv_path):
    """Load processed CSV → (acc [T,6,3], ori [T,6,3,3]) tensors."""
    df = pd.read_csv(csv_path)
    timestamps = sorted(df['timestamp'].unique())
    T = len(timestamps)

    acc = np.zeros((T, 6, 3))
    ori = np.zeros((T, 6, 3, 3))

    for t_idx, ts in enumerate(timestamps):
        frame = df[df['timestamp'] == ts]
        for _, row in frame.iterrows():
            imu_id = int(row['imu_id'])
            if imu_id < 6:
                acc[t_idx, imu_id] = [row['accel_x'], row['accel_y'], row['accel_z']]
                ori[t_idx, imu_id] = _euler_to_rotation_matrix(row['roll'], row['pitch'], row['yaw'])

    return torch.from_numpy(acc).float(), torch.from_numpy(ori).float()


# ---------- Model ----------

def load_model(ckpt_path, device):
    """Load pretrained FIP model."""
    model = FIP()
    model.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------- Inference ----------

def infer_poses(csv_path, model, device, body_params):
    """Run FIP online inference on a single CSV → raw poses [T, 15, 3, 3]."""
    acc, ori = csv_to_tensors(csv_path)

    # Root-relative transformation (same as original evaluate.py)
    acc_root = ori[:, 5:].transpose(-1, -2) @ (acc - acc[:, 5:]).unsqueeze(-1)
    acc_root = acc_root.squeeze(-1)
    ori[:, :5] = ori[:, 5:].transpose(-1, -2) @ ori[:, :5]

    acc_input = acc_root.reshape(-1, 18).to(device) / 30
    ori_input = ori.flatten(1).to(device)
    body_p = body_params[0].unsqueeze(0).to(device)
    pose_ini = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 15, 1, 1).to(device)

    with torch.no_grad():
        model.reset(pose_ini.reshape(1, 15, 3, 3), body_p)
        poses = []
        integ_hc = hip_hc = spine_hc = None

        for i in range(len(acc_input)):
            rot, _, _, _, integ_hc, spine_hc, hip_hc = model.forward_online(
                acc_input[i:i+1].unsqueeze(0),
                ori_input[i:i+1].unsqueeze(0),
                integ_hc=integ_hc, hip_hc=hip_hc, spine_hc=spine_hc,
            )
            poses.append(rot.squeeze())

    return torch.stack(poses).reshape(-1, 15, 3, 3).cpu()


# ---------- T-pose calibration ----------

def calibrate_tpose(raw_poses, tpose_ref, is_tpose=False):
    """Apply T-pose calibration.

    For the T-pose recording itself (is_tpose=True): force all frames to identity.
    For other motions: R_calibrated = R_tpose^{-1} @ R_motion.
    """
    if is_tpose:
        T, J = raw_poses.shape[:2]
        return torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, J, -1, -1).clone()

    tpose_inv = tpose_ref.transpose(-1, -2)   # [15, 3, 3]
    return torch.einsum('jkl,ijlm->ijkm', tpose_inv, raw_poses)


# ---------- Public API ----------

def process_all_motions(csv_dir, output_dir, model, device, body_params, motion_ids, tpose_id='m1'):
    """Full inference + calibration for all motions.

    Returns dict {motion_id: output_path}.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pass 1: infer raw poses for every motion
    raw_poses = {}
    for mid in motion_ids:
        csv_path = os.path.join(csv_dir, f'{mid}_processed.csv')
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping")
            continue
        print(f"  Inferring {mid} ...", end=' ')
        raw_poses[mid] = infer_poses(csv_path, model, device, body_params)
        print(f"{raw_poses[mid].shape[0]} frames")

    if tpose_id not in raw_poses:
        raise RuntimeError(f"T-pose motion '{tpose_id}' not found in {list(raw_poses.keys())}")

    # Pass 2: calibrate (T-pose reference = first frame of tpose recording)
    tpose_ref = raw_poses[tpose_id][0]  # [15, 3, 3]
    results = {}
    for mid, poses in raw_poses.items():
        is_tpose = (mid == tpose_id)
        calibrated = calibrate_tpose(poses, tpose_ref, is_tpose=is_tpose)

        out_path = os.path.join(output_dir, f'{mid}_calibrated.pt')
        torch.save({
            'poses': calibrated,
            'num_frames': calibrated.shape[0],
            'calibrated': True,
        }, out_path)
        results[mid] = out_path

        label = "T-pose (forced identity)" if is_tpose else "calibrated"
        print(f"  {mid}: {label} → {os.path.basename(out_path)}")

    return results
