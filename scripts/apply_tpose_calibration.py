"""
应用T-pose校准到所有动作数据
将m1作为T-pose基准，对m2-m7进行校准，生成最终的校准后姿态数据
这样渲染脚本可以直接使用校准后的数据，更加独立
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
from tqdm import tqdm

POSE_DIR = os.path.join(project_root, 'mydata/processed_final')
OUTPUT_DIR = os.path.join(project_root, 'mydata/processed_final')

def load_tpose_reference():
    """加载m1的T-pose参考数据"""
    tpose_file = os.path.join(POSE_DIR, 'm1_poses.pt')
    
    if not os.path.exists(tpose_file):
        raise FileNotFoundError(f"T-pose文件不存在: {tpose_file}")
    
    data = torch.load(tpose_file, weights_only=False)
    tpose_poses = data['poses']  # [T, 15, 3, 3]
    
    # 使用第一帧作为T-pose参考（或者取平均）
    tpose_ref = tpose_poses[0]  # [15, 3, 3]
    
    print(f"T-pose参考数据已加载 (来自m1第一帧)")
    print(f"  关节数: {tpose_ref.shape[0]}")
    return tpose_ref

def apply_tpose_calibration(poses, tpose_ref):
    """
    应用T-pose校准
    将动作姿态相对于T-pose进行校准
    
    poses: [T, 15, 3, 3] - 原始姿态
    tpose_ref: [15, 3, 3] - T-pose参考
    返回: [T, 15, 3, 3] - 校准后的姿态
    """
    # 计算相对于T-pose的旋转
    # R_calibrated = R_tpose^T @ R_motion
    tpose_inv = tpose_ref.transpose(-1, -2)  # [15, 3, 3]
    calibrated_poses = torch.einsum('jkl,ijlm->ijkm', tpose_inv, poses)
    
    return calibrated_poses

def create_standard_tpose(num_frames, num_joints=15):
    """创建标准T-pose（所有关节旋转为单位矩阵）"""
    # 标准T-pose：所有关节旋转矩阵都是单位矩阵
    standard_tpose = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(num_frames, num_joints, 1, 1)
    return standard_tpose

def process_motion(motion_id, tpose_ref):
    """处理单个动作文件"""
    input_file = os.path.join(POSE_DIR, f'{motion_id}_poses.pt')
    output_file = os.path.join(OUTPUT_DIR, f'{motion_id}_poses_calibrated.pt')
    
    if not os.path.exists(input_file):
        print(f"警告: {input_file} 不存在，跳过...")
        return False
    
    print(f"\n处理 {motion_id}...")
    
    # 加载原始数据
    data = torch.load(input_file, weights_only=False)
    poses = data['poses']
    
    print(f"  原始帧数: {poses.shape[0]}")
    
    # m1强制使用标准T-pose（单位矩阵），不使用推理结果
    if motion_id == 'm1':
        print("  *** 强制设置为标准T-pose（单位矩阵）***")
        calibrated_poses = create_standard_tpose(poses.shape[0], poses.shape[1])
    else:
        print("  应用T-pose校准...")
        calibrated_poses = apply_tpose_calibration(poses, tpose_ref)
    
    # 保存校准后的数据
    output_data = {
        'poses': calibrated_poses,
        'acc': data.get('acc', None),
        'ori': data.get('ori', None),
        'num_frames': calibrated_poses.shape[0],
        'calibrated': True,
        'tpose_reference': 'm1'
    }
    
    torch.save(output_data, output_file)
    print(f"  ✓ 已保存到: {output_file}")
    
    return True

def main():
    print("="*70)
    print("T-pose校准处理")
    print("="*70)
    print("将m1作为T-pose基准，对m2-m7进行校准")
    print("生成独立的校准后姿态数据文件")
    print("="*70)
    
    # 加载T-pose参考
    print("\n加载T-pose参考数据...")
    tpose_ref = load_tpose_reference()
    
    # 处理所有动作
    motion_ids = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
    
    success_count = 0
    for motion_id in motion_ids:
        if process_motion(motion_id, tpose_ref):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"✓ 校准完成! {success_count}/{len(motion_ids)} 个文件处理成功")
    print("="*70)
    print("\n生成的校准后文件:")
    for motion_id in motion_ids:
        output_file = os.path.join(OUTPUT_DIR, f'{motion_id}_poses_calibrated.pt')
        if os.path.exists(output_file):
            print(f"  ✓ {output_file}")
    
    print("\n下一步: 使用渲染脚本渲染校准后的数据")

if __name__ == "__main__":
    main()
