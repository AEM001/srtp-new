#!/usr/bin/env python3
"""Export SMPL parameters from calibrated pose files.

Reads *_calibrated.pt produced by the pipeline and exports standard SMPL
parameters (pose in axis-angle, rotation matrices, joint positions, vertices)
to .npz files.

Usage:
    python export_smpl_params.py                          # Export all motions
    python export_smpl_params.py --motions m2 m3          # Export specific motions
    python export_smpl_params.py --output_dir /tmp/smpl   # Custom output directory
    python export_smpl_params.py --formats pose_aa pose_rotmat joints vertices  # Select outputs
    python export_smpl_params.py --list-formats           # Show all available formats

Available formats:
    pose_aa      : SMPL local pose as axis-angle          [T, 24, 3]
    pose_rotmat  : SMPL local pose as rotation matrices   [T, 24, 3, 3]
    pose_global  : 15-joint global (calibrated) rotations [T, 15, 3, 3]
    joints       : Joint positions from forward kinematics [T, 24, 3]
    vertices     : Mesh vertex positions                   [T, 6890, 3]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

from config import MOTION_IDS, DEVICE, SMPL_MODEL, OUTPUT_DIR
from src.kinematic_model import ParametricModel
from src.eval_tools import glb2local
from model.math import rotation_matrix_to_axis_angle

ALL_FORMATS = ['pose_aa', 'pose_rotmat', 'pose_global', 'joints', 'vertices']

ZERO_JOINT_INDICES = [0, 7, 8, 10, 11, 20, 21, 22, 23]


def load_calibrated_poses(pose_path, device):
    """Load a *_calibrated.pt file → [T, 15, 3, 3] tensor."""
    data = torch.load(pose_path, map_location=device, weights_only=False)
    return data['poses']


def global_to_local_24(poses_global):
    """Convert [T, 15, 3, 3] global poses → [T, 24, 3, 3] local rotation matrices.

    Uses the same glb2local + zero-out logic as the renderer.
    """
    poses_local = glb2local(poses_global)
    poses_local[:, ZERO_JOINT_INDICES] = torch.eye(3, device=poses_local.device)
    return poses_local


def compute_joints(body_model, poses_local, device, batch_size=64):
    """Compute joint positions [T, 24, 3] from local poses via forward kinematics."""
    all_joints = []
    with torch.no_grad():
        for i in range(0, len(poses_local), batch_size):
            batch = poses_local[i:i + batch_size].to(device)
            _, joint_pos = body_model.forward_kinematics(batch, calc_mesh=False)
            all_joints.append(joint_pos.cpu())
    return torch.cat(all_joints, dim=0)


def compute_vertices(body_model, poses_local, device, batch_size=16):
    """Compute mesh vertices [T, V, 3] from local poses via forward kinematics."""
    all_verts = []
    with torch.no_grad():
        for i in range(0, len(poses_local), batch_size):
            batch = poses_local[i:i + batch_size].to(device)
            _, _, verts = body_model.forward_kinematics(batch, calc_mesh=True)
            all_verts.append(verts.cpu())
    return torch.cat(all_verts, dim=0)


def export_motion(pose_path, output_path, formats, body_model, device):
    """Export SMPL parameters for a single motion.

    Returns dict of exported format names → array shapes.
    """
    poses_global = load_calibrated_poses(pose_path, device)
    T = poses_global.shape[0]

    result = {}

    # Always compute local poses if any derived format is requested
    need_local = any(f in formats for f in ['pose_aa', 'pose_rotmat', 'joints', 'vertices'])
    poses_local = global_to_local_24(poses_global) if need_local else None

    if 'pose_global' in formats:
        arr = poses_global.cpu().numpy()
        result['pose_global'] = arr
        print(f"    pose_global:  {arr.shape}")

    if 'pose_rotmat' in formats:
        arr = poses_local.cpu().numpy()
        result['pose_rotmat'] = arr
        print(f"    pose_rotmat:  {arr.shape}")

    if 'pose_aa' in formats:
        # [T, 24, 3, 3] → [T*24, 3, 3] → [T*24, 3] → [T, 24, 3]
        flat = poses_local.reshape(-1, 3, 3)
        aa = rotation_matrix_to_axis_angle(flat).reshape(T, 24, 3)
        arr = aa.cpu().numpy()
        result['pose_aa'] = arr
        print(f"    pose_aa:      {arr.shape}")

    if 'joints' in formats:
        joints = compute_joints(body_model, poses_local, device)
        arr = joints.numpy()
        result['joints'] = arr
        print(f"    joints:       {arr.shape}")

    if 'vertices' in formats:
        verts = compute_vertices(body_model, poses_local, device)
        arr = verts.numpy()
        result['vertices'] = arr
        print(f"    vertices:     {arr.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **result, num_frames=T)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    → saved: {output_path}  ({file_size:.1f} MB)")

    return {k: v.shape for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser(
        description='Export SMPL parameters from calibrated pose files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--motions', nargs='+', default=MOTION_IDS,
        help=f'Motion IDs to export (default: {MOTION_IDS})',
    )
    parser.add_argument(
        '--formats', nargs='+', default=['pose_aa', 'pose_rotmat', 'joints'],
        choices=ALL_FORMATS,
        help='Which SMPL data to include (default: pose_aa pose_rotmat joints)',
    )
    parser.add_argument(
        '--pose_dir', default=None,
        help=f'Directory containing *_calibrated.pt files (default: <output>/poses)',
    )
    parser.add_argument(
        '--output_dir', default=None,
        help=f'Output directory for .npz files (default: <output>/smpl_params)',
    )
    parser.add_argument(
        '--list-formats', action='store_true',
        help='Print available formats and exit',
    )
    args = parser.parse_args()

    if args.list_formats:
        print("Available export formats:")
        print("  pose_aa      : SMPL local pose as axis-angle          [T, 24, 3]")
        print("  pose_rotmat  : SMPL local pose as rotation matrices   [T, 24, 3, 3]")
        print("  pose_global  : 15-joint global (calibrated) rotations [T, 15, 3, 3]")
        print("  joints       : Joint positions from forward kinematics [T, 24, 3]")
        print("  vertices     : Mesh vertex positions                   [T, 6890, 3]")
        return

    pose_dir = args.pose_dir or os.path.join(OUTPUT_DIR, 'poses')
    output_dir = args.output_dir or os.path.join(OUTPUT_DIR, 'smpl_params')

    print("=" * 60)
    print("SMPL Parameter Export")
    print("=" * 60)
    print(f"  Device:     {DEVICE}")
    print(f"  Motions:    {args.motions}")
    print(f"  Formats:    {args.formats}")
    print(f"  Pose dir:   {pose_dir}")
    print(f"  Output dir: {output_dir}")

    # Load SMPL body model only if joints or vertices are requested
    need_body_model = any(f in args.formats for f in ['joints', 'vertices'])
    body_model = None
    if need_body_model:
        print(f"\n  Loading SMPL model: {SMPL_MODEL}")
        body_model = ParametricModel(SMPL_MODEL, device=DEVICE)

    t0 = time.time()
    exported = 0
    skipped = 0

    for mid in args.motions:
        pose_path = os.path.join(pose_dir, f'{mid}_calibrated.pt')
        output_path = os.path.join(output_dir, f'{mid}_smpl.npz')

        if not os.path.exists(pose_path):
            print(f"\n  [{mid}] Warning: {pose_path} not found, skipping")
            skipped += 1
            continue

        print(f"\n  [{mid}]")
        export_motion(pose_path, output_path, args.formats, body_model, DEVICE)
        exported += 1

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done: {exported} exported, {skipped} skipped  ({elapsed:.1f}s)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
