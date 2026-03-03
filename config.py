"""Central configuration for the FIP motion reconstruction pipeline."""
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Paths ---
SMPL_MODEL = os.path.join(PROJECT_ROOT, 'data', 'SMPL_male.pkl')
MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, 'ckpt', 'best_model.pt')
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# --- Motion IDs ---
MOTION_IDS = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
TPOSE_ID = 'm1'

# --- IMU ---
IMU_SAMPLE_RATE = 50  # Hz
BROKEN_IMU_ID = 0
FILL_IMU_ID = 3

# --- Rendering ---
RENDER_FPS = 30
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

# --- Device ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Body parameters (from DIP-IMU dataset, scaled by /100) ---
BODY_PARAMS = torch.tensor([
    [0, 186, 84.24, 55.08], [0, 178, 80.62, 52.71],
    [0, 187, 84.70, 55.37], [0, 170, 77.00, 50.34],
    [0, 180, 81.53, 53.30], [100, 172, 78.60, 51.98],
    [0, 178, 80.62, 52.71], [0, 180, 81.53, 53.30],
    [0, 187, 84.70, 55.37], [0, 181, 77.00, 50.34],
]) / 100

# --- Coordinate transform: IMU (X-up, Y-left, Z-front) → Model (X-left, Y-up, Z-front) ---
import numpy as np
COORD_TRANSFORM = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
])
