"""
对新的IMU数据运行动作重建推理
处理m1-m7的数据
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

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
            if imu_id < 6:  # 确保ID在范围内
                acc_data[t_idx, imu_id] = [row['accel_x'], row['accel_y'], row['accel_z']]
                ori_data[t_idx, imu_id] = euler_to_rotation_matrix(
                    row['roll'], row['pitch'], row['yaw']
                )
    
    return torch.from_numpy(acc_data).float(), torch.from_numpy(ori_data).float()

def run_inference(csv_file, model, device, body_parm, output_name):
    """Run model inference on CSV data"""
    print(f"\n{'='*60}")
    print(f"处理: {csv_file}")
    print(f"{'='*60}")
    
    # Load data
    acc, ori = load_csv_to_tensors(csv_file)
    
    print(f"数据形状: acc={acc.shape}, ori={ori.shape}")
    print(f"帧数: {acc.shape[0]}")
    print(f"时长: {acc.shape[0]/50:.2f} 秒 (50Hz)")
    
    # Check gravity direction (should be close to 1g in Y-axis)
    avg_acc = acc.mean(dim=(0, 1))
    print(f"平均加速度: ({avg_acc[0]:.3f}, {avg_acc[1]:.3f}, {avg_acc[2]:.3f})")
    if avg_acc[1] > 0.7:
        print(f"  ✓ Y轴重力检查通过: {avg_acc[1]:.3f}g")
    else:
        print(f"  ⚠ 警告: Y轴重力 = {avg_acc[1]:.3f}g")
    
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
    print("运行模型推理...")
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
    
    print(f"推理完成! 输出形状: {pose_p.shape}")
    
    # Save results
    output_path = f'mydata/processed_final/{output_name}_poses.pt'
    torch.save({
        'poses': pose_p,
        'acc': acc,
        'ori': ori,
        'num_frames': b
    }, output_path)
    print(f"结果保存到: {output_path}")
    
    return pose_p

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("\n坐标系统: X-左, Y-上, Z-前")
    print("重力应在Y方向 (~1.0g)\n")
    
    # Body parameters
    body_parm = [[0,186,84.24332236,55.0763866],[0,178, 80.61995365, 52.70750976],
                 [0,187, 84.69624344, 55.37249621],[0,170, 76.99658495, 50.33863291],
                 [0,180, 81.52579583, 53.29972897],[100,172, 78.59580262, 51.98239681],
                 [0,178, 80.61995365, 52.70750976],[0,180, 81.52579583, 53.29972897],
                 [0,187, 84.69624344, 55.37249621],[0,181, 76.99658495, 50.33863291]]
    body_parm = torch.tensor(body_parm) / 100
    
    # Load model
    print("加载模型...")
    model = FIP()
    model.load_state_dict(torch.load('ckpt/best_model.pt', weights_only=False, map_location=device))
    model.to(device)
    model.eval()
    print("模型加载成功!\n")
    
    # Process files m1-m7
    files = [(f'mydata/processed_final/m{i}_final.csv', f'm{i}') for i in range(1, 8)]
    
    results = {}
    for csv_file, name in files:
        if os.path.exists(csv_file):
            poses = run_inference(csv_file, model, device, body_parm, name)
            results[name] = poses
        else:
            print(f"警告: {csv_file} 未找到")
    
    print("\n" + "="*60)
    print("推理完成!")
    print("="*60)
    print("\n生成的文件:")
    for name in results.keys():
        print(f"  - mydata/processed_final/{name}_poses.pt")
    
    print("\n下一步: 运行可视化脚本")

if __name__ == "__main__":
    main()
