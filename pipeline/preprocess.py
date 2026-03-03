"""Preprocess raw ODT IMU recordings into model-ready CSV format.

Merges the old extract_odt_data.py + process_new_imu_data.py into one step.
Pipeline: ODT → parse → fill broken IMU → coordinate transform → CSV
"""
import os
import re
import numpy as np
import pandas as pd
from odf import text, teletype
from odf.opendocument import load

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COORD_TRANSFORM, BROKEN_IMU_ID, FILL_IMU_ID

# ---------- ODT parsing ----------

_DATA_PATTERN = re.compile(
    r'\s*([\d.]+)\s*\|\s*(\d+)\s*\|'
    r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\|'
    r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'
)


def extract_imu_records(odt_path):
    """Extract IMU data records directly from an ODT file."""
    doc = load(odt_path)
    records = []
    for para in doc.getElementsByType(text.P):
        line = teletype.extractText(para).strip()
        m = _DATA_PATTERN.match(line)
        if m:
            records.append({
                'timestamp': float(m.group(1)),
                'imu_id':    int(m.group(2)),
                'accel_x':   float(m.group(3)),
                'accel_y':   float(m.group(4)),
                'accel_z':   float(m.group(5)),
                'roll':      float(m.group(6)),
                'pitch':     float(m.group(7)),
                'yaw':       float(m.group(8)),
            })
    return records


# ---------- IMU processing helpers ----------

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


def _rotation_matrix_to_euler(R):
    """3x3 rotation matrix → Euler angles (degrees)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# ---------- Core processing ----------

def process_records(records):
    """Process raw IMU records: fill broken sensor + coordinate transform → DataFrame."""
    df = pd.DataFrame(records)
    T = COORD_TRANSFORM

    # Fill broken IMU with substitute
    filled = []
    for ts in df['timestamp'].unique():
        frame = df[df['timestamp'] == ts]
        imu_map = {int(r['imu_id']): r for _, r in frame.iterrows()}
        if BROKEN_IMU_ID not in imu_map and FILL_IMU_ID in imu_map:
            row0 = imu_map[FILL_IMU_ID].copy()
            row0['imu_id'] = BROKEN_IMU_ID
            filled.append(row0)
        for imu_id in sorted(imu_map):
            filled.append(imu_map[imu_id])
    df = pd.DataFrame(filled).sort_values(['timestamp', 'imu_id']).reset_index(drop=True)

    # Apply coordinate transform
    transformed = []
    for _, row in df.iterrows():
        accel = T @ np.array([row['accel_x'], row['accel_y'], row['accel_z']])
        R = T @ _euler_to_rotation_matrix(row['roll'], row['pitch'], row['yaw']) @ T.T
        roll_new, pitch_new, yaw_new = _rotation_matrix_to_euler(R)
        transformed.append({
            'timestamp': row['timestamp'],
            'imu_id':    int(row['imu_id']),
            'accel_x':   accel[0],
            'accel_y':   accel[1],
            'accel_z':   accel[2],
            'roll':      roll_new,
            'pitch':     pitch_new,
            'yaw':       yaw_new,
        })

    return pd.DataFrame(transformed)


# ---------- Public API ----------

def preprocess_motion(odt_path, output_csv):
    """Full preprocessing pipeline: ODT → processed CSV (single motion)."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    records = extract_imu_records(odt_path)
    if not records:
        print(f"  Warning: no IMU data in {odt_path}")
        return None

    df = process_records(records)
    df.to_csv(output_csv, index=False)

    duration = df['timestamp'].max() - df['timestamp'].min()
    n_frames = len(df['timestamp'].unique())
    print(f"  {os.path.basename(odt_path)}: {n_frames} frames, {duration:.1f}s → {os.path.basename(output_csv)}")
    return df
