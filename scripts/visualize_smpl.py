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
import trimesh
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
    """使用trimesh渲染mesh序列为视频（Mac兼容）"""
    frames = []
    
    # 创建场景
    scene = trimesh.Scene()
    
    # 设置相机参数
    resolution = (640, 480)
    fov = (60, 45)  # 水平和垂直视场角
    
    for i, verts in enumerate(tqdm(vertices_list, desc="渲染帧")):
        # 创建mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.visual.vertex_colors = [100, 150, 200, 255]
        
        # 更新场景
        scene = trimesh.Scene()
        scene.add_geometry(mesh)
        
        # 设置相机位置（从正面看）
        camera_transform = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 2.5],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.camera_transform = camera_transform
        
        # 渲染为图像
        try:
            # 使用trimesh的渲染功能
            data = scene.save_image(resolution=resolution, visible=False)
            from PIL import Image
            import io
            image = np.array(Image.open(io.BytesIO(data)))
            frames.append(image[:, :, :3])  # 只取RGB
        except Exception as e:
            print(f"渲染帧 {i} 失败: {e}")
            # 创建空白帧
            frames.append(np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 255)
    
    # 保存视频
    writer = imageio.get_writer(output_path, fps=fps, format='FFMPEG', codec='libx264')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    return frames

def load_tpose_calibration(m, device):
    """
    加载m1的T-pose数据作为校准基准
    返回T-pose的旋转矩阵
    """
    tpose_file = os.path.join(POSE_DIR, 'm1_poses.pt')
    if not os.path.exists(tpose_file):
        print("警告: m1_poses.pt不存在，无法进行T-pose校准")
        return None
    
    data = torch.load(tpose_file, map_location=device, weights_only=False)
    tpose_poses = data['poses']  # [T, 15, 3, 3]
    
    # 使用第一帧作为T-pose参考（或者取平均）
    tpose_ref = tpose_poses[0]  # [15, 3, 3]
    
    print(f"T-pose校准数据已加载 (来自m1第一帧)")
    return tpose_ref

def apply_tpose_calibration(poses, tpose_ref):
    """
    应用T-pose校准
    将动作姿态相对于T-pose进行校准
    poses: [T, 15, 3, 3]
    tpose_ref: [15, 3, 3]
    """
    if tpose_ref is None:
        return poses
    
    # 计算相对于T-pose的旋转
    # R_calibrated = R_tpose^T @ R_motion
    tpose_inv = tpose_ref.transpose(-1, -2)  # [15, 3, 3]
    calibrated_poses = torch.einsum('jkl,ijlm->ijkm', tpose_inv, poses)
    
    return calibrated_poses

def poses_to_smpl_vertices(poses, m, device='cpu', tpose_ref=None, apply_calibration=True):
    """将姿态旋转矩阵转换为SMPL顶点"""
    T = poses.shape[0]
    
    # 应用T-pose校准（对于m2-m7）
    if apply_calibration and tpose_ref is not None:
        poses = apply_tpose_calibration(poses, tpose_ref)
    
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

def visualize_motion(motion_id, m, faces, device, tpose_ref=None):
    """可视化单个动作"""
    pose_file = os.path.join(POSE_DIR, f'{motion_id}_poses.pt')
    
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
    
    # m1是T-pose，不需要校准；m2-m7需要基于m1校准
    apply_calibration = (motion_id != 'm1')
    if apply_calibration:
        print("应用T-pose校准 (基于m1)")
    else:
        print("T-pose参考数据，不进行校准")
    
    # 转换为SMPL顶点
    vertices_list = poses_to_smpl_vertices(
        poses, m, device, 
        tpose_ref=tpose_ref, 
        apply_calibration=apply_calibration
    )
    
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
    
    # 加载T-pose校准数据
    print("\n加载T-pose校准数据...")
    tpose_ref = load_tpose_calibration(m, device)
    
    # 渲染所有动作
    generated_videos = []
    
    for motion_id in MOTION_IDS:
        video_path = visualize_motion(motion_id, m, faces, device, tpose_ref)
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
