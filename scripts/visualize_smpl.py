"""
SMPL人体模型渲染脚本
使用pyrender进行真实人体mesh渲染
需要在d2l conda环境下运行

重要：m1是T-pose校准数据，m2-m7的动作都基于m1的T-pose进行校准
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
import imageio

from src import kinematic_model
from src.eval_tools import glb2local

# 配置
SMPL_FILE = os.path.join(project_root, 'data/SMPL_male.pkl')
POSE_DIR = os.path.join(project_root, 'mydata/processed_final')
OUTPUT_DIR = os.path.join(project_root, 'mydata/processed_final')

# 要渲染的动作 (m1是T-pose，m2-m7是动作)
MOTION_IDS = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
FPS = 30

def render_mesh_sequence(vertices_list, faces, output_path, fps=30):
    """使用matplotlib渲染mesh序列为视频（Mac兼容）"""
    frames = []
    
    # 旋转顶点：人体从横躺（正面朝Z轴）变为直立（正面朝Y轴）
    # 绕X轴旋转-90度，让Z轴变成Y轴
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],  # Y_new = -Z_old
        [0, 1, 0]    # Z_new = Y_old
    ])
    
    rotated_vertices_list = []
    for verts in vertices_list:
        rotated_verts = verts @ rotation_matrix.T
        rotated_vertices_list.append(rotated_verts)
    
    vertices_list = rotated_vertices_list
    
    # 计算所有帧的边界
    all_verts = np.concatenate(vertices_list, axis=0)
    x_center = (all_verts[:, 0].min() + all_verts[:, 0].max()) / 2
    y_center = (all_verts[:, 1].min() + all_verts[:, 1].max()) / 2
    z_center = (all_verts[:, 2].min() + all_verts[:, 2].max()) / 2
    max_range = max(
        all_verts[:, 0].max() - all_verts[:, 0].min(),
        all_verts[:, 1].max() - all_verts[:, 1].min(),
        all_verts[:, 2].max() - all_verts[:, 2].min()
    ) / 2 * 1.2
    
    fig = plt.figure(figsize=(8, 6), dpi=100)
    
    for i, verts in enumerate(tqdm(vertices_list, desc="渲染帧")):
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        
        # 简化渲染：只绘制部分面（每隔N个面绘制一个）
        step = max(1, len(faces) // 2000)  # 最多绘制2000个面
        selected_faces = faces[::step]
        
        # 创建多边形集合
        mesh_faces = []
        for face in selected_faces:
            triangle = [verts[face[0]], verts[face[1]], verts[face[2]]]
            mesh_faces.append(triangle)
        
        # 添加mesh
        mesh_collection = Poly3DCollection(mesh_faces, alpha=0.9)
        mesh_collection.set_facecolor([0.4, 0.6, 0.8])
        mesh_collection.set_edgecolor([0.3, 0.5, 0.7])
        mesh_collection.set_linewidth(0.1)
        ax.add_collection3d(mesh_collection)
        
        # 设置坐标轴范围
        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_zlim(z_center - max_range, z_center + max_range)
        
        # 设置视角：从前方水平看直立的人体
        # 旋转后人体Z轴向上，Y轴向前，从Y轴正方向看（正面朝向观察者）
        ax.view_init(elev=0, azim=90)
        
        # 隐藏坐标轴
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        
        # 保存帧
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())
    
    plt.close(fig)
    
    # 保存视频
    writer = imageio.get_writer(output_path, fps=fps, format='FFMPEG', codec='libx264')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    return frames

def poses_to_smpl_vertices(poses, m, device='cpu'):
    """将姿态旋转矩阵转换为SMPL顶点"""
    T = poses.shape[0]
    
    # 转换为local poses
    poses_local = glb2local(poses)
    
    # 生成顶点
    vertices_list = []
    
    with torch.no_grad():
        for i in tqdm(range(T), desc="生成SMPL顶点"):
            pose = poses_local[i:i+1].to(device)
            # 设置某些关节为单位矩阵
            pose[:, [0, 7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=device)
            _, _, mesh = m.forward_kinematics(pose, calc_mesh=True)
            vertices_list.append(mesh[0].cpu().numpy())
    
    return vertices_list

def visualize_motion(motion_id, m, faces, device):
    """可视化单个动作"""
    # 使用校准后的数据
    pose_file = os.path.join(POSE_DIR, f'{motion_id}_poses_calibrated.pt')
    
    if not os.path.exists(pose_file):
        print(f"警告: {pose_file} 不存在，跳过...")
        return None
    
    print(f"\n{'='*60}")
    print(f"处理: {motion_id}")
    print(f"{'='*60}")
    
    # 加载姿态数据
    data = torch.load(pose_file, map_location=device, weights_only=False)
    poses = data['poses']
    
    print(f"帧数: {poses.shape[0]}")
    print(f"关节数: {poses.shape[1]}")
    print(f"已校准: {data.get('calibrated', False)}")
    
    # 直接使用校准后的数据，不需要再次校准
    vertices_list = poses_to_smpl_vertices(poses, m, device)
    
    # 渲染视频
    output_path = os.path.join(OUTPUT_DIR, f'{motion_id}_animation.mp4')
    print(f"渲染到: {output_path}")
    frames = render_mesh_sequence(vertices_list, faces, output_path, fps=FPS)
    
    print(f"✓ 完成: {output_path} ({len(frames)}帧)")
    return output_path

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("SMPL人体模型渲染")
    print("="*60)
    print(f"设备: {device}")
    print(f"动作列表: {MOTION_IDS}")
    print(f"m1 = T-pose校准, m2-m7 = 动作序列")
    print("="*60)
    
    # 加载SMPL模型
    print("\n加载SMPL模型...")
    m = kinematic_model.ParametricModel(SMPL_FILE, device=device)
    faces = m.face
    print(f"SMPL模型已加载 (面数: {len(faces)})")
    
    # 渲染所有动作（使用校准后的数据）
    generated_videos = []
    
    for motion_id in MOTION_IDS:
        video_path = visualize_motion(motion_id, m, faces, device)
        if video_path:
            generated_videos.append(video_path)
    
    print("\n" + "="*60)
    print("✓ 全部完成!")
    print("="*60)
    print("\n生成的视频:")
    for video in generated_videos:
        print(f"  ✓ {video}")

if __name__ == "__main__":
    main()
