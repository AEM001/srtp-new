"""Central configuration for the FIP motion reconstruction pipeline."""
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Paths ---
SMPL_MODEL = os.path.join(PROJECT_ROOT, 'data', 'SMPL_male.pkl')
MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, 'ckpt', 'best_model.pt')
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# --- Motion IDs (offline pipeline) ---
MOTION_IDS = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
TPOSE_ID = 'm1'

# --- IMU ---
IMU_SAMPLE_RATE = 50  # Hz
BROKEN_IMU_ID = 0
FILL_IMU_ID = 3

# --- Rendering ---
RENDER_FPS = int(os.environ.get('RENDER_FPS', '30'))
RENDER_WIDTH = int(os.environ.get('RENDER_WIDTH', '640'))
RENDER_HEIGHT = int(os.environ.get('RENDER_HEIGHT', '480'))

# --- OpenGL backend (egl for GPU/headless, osmesa for CPU-only Docker) ---
# Override via env: PYOPENGL_PLATFORM=egl or PYOPENGL_PLATFORM=osmesa
OPENGL_PLATFORM = os.environ.get('PYOPENGL_PLATFORM', 'osmesa')

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

# --- Default body params for real-time streaming (gender=male, height=178cm, weight=80.6kg) ---
DEFAULT_BODY_PARAMS = torch.tensor([0, 178, 80.62, 52.71]) / 100

# --- Coordinate transform: IMU (X-up, Y-left, Z-front) → Model (X-left, Y-up, Z-front) ---
import numpy as np
COORD_TRANSFORM = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
])

# --- Real-time streaming server ---
IMU_HOST = os.environ.get('IMU_HOST', '0.0.0.0')
IMU_PORT = int(os.environ.get('IMU_PORT', '9000'))
STREAM_HOST = os.environ.get('STREAM_HOST', '0.0.0.0')
STREAM_PORT = int(os.environ.get('STREAM_PORT', '8080'))
STREAM_JPEG_QUALITY = int(os.environ.get('STREAM_JPEG_QUALITY', '85'))
