"""
Run model inference on processed CSV data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'human_body_prior'))

import torch
import numpy as np
import pandas as pd
from model.net import FIP
from src.eval_tools import *

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (degrees) to rotation matrix"""
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx

def load_csv_to_tensors(csv_file):
    """Load CSV file and convert to model input format"""
    df = pd.read_csv(csv_file)
    
    # Get unique timestamps
    timestamps = sorted(df['timestamp'].unique())
    T = len(timestamps)
    
    # Initialize tensors
    acc_data = np.zeros((T, 6, 3))
    ori_data = np.zeros((T, 6, 3, 3))
    
    for t_idx, timestamp in enumerate(timestamps):
        frame_data = df[df['timestamp'] == timestamp]
        
        for _, row in frame_data.iterrows():
            imu_id = int(row['imu_id'])
            acc_data[t_idx, imu_id] = [row['accel_x'], row['accel_y'], row['accel_z']]
            ori_data[t_idx, imu_id] = euler_to_rotation_matrix(
                row['roll'], row['pitch'], row['yaw']
            )
    
    return torch.from_numpy(acc_data).float(), torch.from_numpy(ori_data).float()

def run_inference(csv_file, model, device, body_parm, output_name):
    """Run model inference on CSV data"""
    print(f"\n{'='*60}")
    print(f"Processing: {csv_file}")
    print(f"{'='*60}")
    
    # Load data
    acc, ori = load_csv_to_tensors(csv_file)
    
    print(f"Data shape: acc={acc.shape}, ori={ori.shape}")
    print(f"Number of frames: {acc.shape[0]}")
    print(f"Duration: {acc.shape[0]/50:.2f} seconds (50Hz)")
    
    # Check gravity direction (should be close to -1g in Y-axis)
    avg_acc = acc.mean(dim=(0, 1))
    print(f"Average acceleration: ({avg_acc[0]:.3f}, {avg_acc[1]:.3f}, {avg_acc[2]:.3f})")
    print(f"  → Gravity check: Y-axis should be ~-1.0g: {avg_acc[1]:.3f}g ✓" if avg_acc[1] < -0.8 else f"  → Warning: Y-axis gravity = {avg_acc[1]:.3f}g")
    
    # Prepare data for model
    acc_root = ori[:, 5:].transpose(-1, -2) @ (acc - acc[:, 5:]).unsqueeze(-1)
    acc_root = acc_root.squeeze(-1)
    ori[:, :5] = ori[:, 5:].transpose(-1, -2) @ ori[:, :5]
    
    acc_input = acc_root.reshape(-1, 18).to(device) / 30
    ori_input = ori.flatten(1).to(device)
    
    # Use default body parameters
    body_p = body_parm[0].unsqueeze(0).to(device)
    
    # Initialize pose
    b = acc_input.size(0)
    pose_ini = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 15, 1, 1).to(device)
    
    # Run inference
    print("Running model inference...")
    with torch.no_grad():
        model.reset(pose_ini.reshape(1, 15, 3, 3), body_p)
        poses_p = []
        
        for i in range(len(acc_input)):
            acc_ = acc_input[i:i+1].unsqueeze(0)
            ori_ = ori_input[i:i+1].unsqueeze(0)
            
            if i == 0:
                integ_hc = None
                hip_hc = None
                spine_hc = None
            
            rot, _, _, _, integ_hc, spine_hc, hip_hc = model.forward_online(
                acc_, ori_, integ_hc=integ_hc, hip_hc=hip_hc, spine_hc=spine_hc
            )
            poses_p.append(rot.squeeze())
        
        pose_p = torch.stack(poses_p).reshape(-1, 15, 3, 3).cpu()
    
    print(f"Inference completed! Output shape: {pose_p.shape}")
    
    # Save results
    output_path = f'mydata/processed_final/{output_name}_poses.pt'
    torch.save({
        'poses': pose_p,
        'acc': acc,
        'ori': ori,
        'num_frames': b
    }, output_path)
    print(f"Results saved to: {output_path}")
    
    return pose_p

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\nCoordinate system: X-right, Y-up, Z-forward")
    print("Gravity should be in -Y direction (~-1.0g)\n")
    
    # Body parameters
    body_parm = [[0,186,84.24332236,55.0763866],[0,178, 80.61995365, 52.70750976],
                 [0,187, 84.69624344, 55.37249621],[0,170, 76.99658495, 50.33863291],
                 [0,180, 81.52579583, 53.29972897],[100,172, 78.59580262, 51.98239681],
                 [0,178, 80.61995365, 52.70750976],[0,180, 81.52579583, 53.29972897],
                 [0,187, 84.69624344, 55.37249621],[0,181, 76.99658495, 50.33863291]]
    body_parm = torch.tensor(body_parm) / 100
    
    # Load model
    print("Loading model...")
    model = FIP()
    model.load_state_dict(torch.load('ckpt/best_model.pt', weights_only=False, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")
    
    # Process files
    files = [
        ('mydata/processed_final/1_final.csv', '1'),
        ('mydata/processed_final/2_final.csv', '2'),
        ('mydata/processed_final/3_final.csv', '3')
    ]
    
    results = {}
    for csv_file, name in files:
        if os.path.exists(csv_file):
            poses = run_inference(csv_file, model, device, body_parm, name)
            results[name] = poses
        else:
            print(f"Warning: {csv_file} not found")
    
    print("\n" + "="*60)
    print("Inference completed!")
    print("="*60)
    print("\nGenerated files:")
    for name in results.keys():
        print(f"  - mydata/processed_final/{name}_poses.pt")
    
    print("\nNext: Run visualization script")

if __name__ == "__main__":
    main()
