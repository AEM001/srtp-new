"""
简化版可视化脚本 - 直接可视化姿态数据
不依赖SMPL模型，适用于Mac环境
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import imageio

# 配置
MOTION_IDS = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
MAX_FRAMES = 100  # 限制帧数加快渲染
FPS = 30

# 骨架连接关系 (15个关节)
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),  # 左腿
    (0, 4), (4, 5), (5, 6),  # 右腿
    (0, 7), (7, 8), (8, 9),  # 脊柱到头
    (8, 10), (10, 11), (11, 12),  # 左臂
    (8, 13), (13, 14), (14, 15)   # 右臂
]

def rotation_matrix_to_position(poses, scale=1.0):
    """
    从旋转矩阵估算关节位置
    poses: [T, 15, 3, 3]
    返回: [T, 15, 3] 关节位置
    """
    T = poses.shape[0]
    positions = np.zeros((T, 15, 3))
    
    # 简单的骨架结构 - 基于旋转矩阵的方向
    bone_lengths = [0.4, 0.4, 0.3,  # 左腿
                   0.4, 0.4, 0.3,  # 右腿
                   0.3, 0.3, 0.2,  # 脊柱
                   0.3, 0.3, 0.2,  # 左臂
                   0.3, 0.3, 0.2]  # 右臂
    
    for t in range(T):
        # 根节点在原点
        positions[t, 0] = [0, 0, 0]
        
        # 简化：使用旋转矩阵的第一列作为方向
        for i in range(1, 15):
            parent = i // 3 if i < 9 else 8
            direction = poses[t, i, :, 0]  # 使用旋转矩阵第一列
            positions[t, i] = positions[t, parent] + direction * bone_lengths[i-1] * scale
    
    return positions

def render_skeleton_video(positions, output_path, fps=30):
    """
    渲染骨架动画
    positions: [T, 15, 3]
    """
    print(f"渲染到 {output_path}...")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算边界
    all_pos = positions.reshape(-1, 3)
    margin = 0.5
    x_range = [all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin]
    y_range = [all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin]
    z_range = [all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin]
    
    frames = []
    
    for t in tqdm(range(len(positions)), desc="渲染帧"):
        ax.clear()
        
        pos = positions[t]
        
        # 绘制关节点
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=50, alpha=0.8)
        
        # 绘制骨骼连接
        for conn in SKELETON_CONNECTIONS:
            if conn[1] < len(pos):
                ax.plot([pos[conn[0], 0], pos[conn[1], 0]],
                       [pos[conn[0], 1], pos[conn[1], 1]],
                       [pos[conn[0], 2], pos[conn[1], 2]],
                       'b-', linewidth=2, alpha=0.7)
        
        # 设置坐标轴
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {t+1}/{len(positions)}')
        ax.view_init(elev=10, azim=45 + t*0.5)  # 旋转视角
        
        # 保存帧
        fig.canvas.draw()
        # 使用buffer_rgba()替代tostring_rgb()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]  # 只取RGB通道
        frames.append(image)
    
    plt.close(fig)
    
    # 保存视频
    writer = imageio.get_writer(output_path, format='FFMPEG', fps=fps, codec='libx264')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    print(f"✓ 视频已保存: {output_path}")

def process_motion(motion_id):
    """处理单个动作"""
    pose_file = os.path.join(project_root, 'mydata', 'processed_final', f'{motion_id}_poses.pt')
    
    if not os.path.exists(pose_file):
        print(f"⚠ {pose_file} 不存在，跳过...")
        return
    
    print(f"\n处理 {motion_id}...")
    
    # 加载姿态数据
    data = torch.load(pose_file, weights_only=False)
    poses = data['poses'].numpy()  # [T, 15, 3, 3]
    
    total_frames = len(poses)
    max_frames = min(total_frames, MAX_FRAMES) if MAX_FRAMES else total_frames
    poses = poses[:max_frames]
    
    print(f"  总帧数: {total_frames}, 渲染帧数: {max_frames}")
    
    # 转换为关节位置
    positions = rotation_matrix_to_position(poses)
    
    # 渲染视频
    output_path = os.path.join(project_root, f'output_{motion_id}.mp4')
    render_skeleton_video(positions, output_path, fps=FPS)

def main():
    print("="*60)
    print("简化版可视化 - 骨架动画")
    print("="*60)
    print(f"动作列表: {MOTION_IDS}")
    print(f"最大帧数: {MAX_FRAMES if MAX_FRAMES else '全部'}")
    print(f"帧率: {FPS} FPS")
    print("="*60)
    
    for motion_id in MOTION_IDS:
        process_motion(motion_id)
    
    print("\n" + "="*60)
    print("✓ 全部完成!")
    print("="*60)

if __name__ == '__main__':
    main()
