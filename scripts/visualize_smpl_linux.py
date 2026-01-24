"""
SMPL人体模型渲染脚本 - Linux服务器版本
使用pyrender + EGL进行离屏渲染
需要在Linux服务器上运行，需要安装: pip install pyrender trimesh imageio imageio-ffmpeg

重要：m1是T-pose校准数据，m2-m7的动作都基于m1的T-pose进行校准
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

# Linux上使用EGL后端
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import numpy as np
import trimesh
import pyrender
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
    """使用pyrender渲染mesh序列为视频（Linux EGL）"""
    
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.3, 0.3, 0.3])
    
    # 相机设置 - 从正面看人体
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # 光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)
    
    # 额外光源
    light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light2_pose = np.eye(4)
    light2_pose[:3, 3] = [0, 1, 2]
    scene.add(light2, pose=light2_pose)
    
    # 渲染器
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    
    frames = []
    mesh_node = None
    
    for i, verts in enumerate(tqdm(vertices_list, desc="渲染帧")):
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.visual.vertex_colors = [100, 150, 200, 255]
        
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        
        if mesh_node is not None:
            scene.remove_node(mesh_node)
        mesh_node = scene.add(py_mesh)
        
        color, _ = r.render(scene)
        frames.append(color)
    
    r.delete()
    
    # 保存视频
    writer = imageio.get_writer(output_path, fps=fps, format='FFMPEG', codec='libx264')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    return frames

def load_tpose_calibration(device):
    """加载m1的T-pose数据作为校准基准"""
    tpose_file = os.path.join(POSE_DIR, 'm1_poses.pt')
    if not os.path.exists(tpose_file):
        print("警告: m1_poses.pt不存在，无法进行T-pose校准")
        return None
    
    data = torch.load(tpose_file, map_location=device, weights_only=False)
    tpose_poses = data['poses']
    tpose_ref = tpose_poses[0]  # 使用第一帧作为T-pose参考
    
    print(f"T-pose校准数据已加载 (来自m1第一帧)")
    return tpose_ref

def apply_tpose_calibration(poses, tpose_ref):
    """应用T-pose校准"""
    if tpose_ref is None:
        return poses
    
    tpose_inv = tpose_ref.transpose(-1, -2)
    calibrated_poses = torch.einsum('jkl,ijlm->ijkm', tpose_inv, poses)
    return calibrated_poses

def poses_to_smpl_vertices(poses, m, device='cpu', tpose_ref=None, apply_calibration=True):
    """将姿态旋转矩阵转换为SMPL顶点"""
    T = poses.shape[0]
    
    if apply_calibration and tpose_ref is not None:
        poses = apply_tpose_calibration(poses, tpose_ref)
    
    poses_local = glb2local(poses)
    
    vertices_list = []
    
    with torch.no_grad():
        for i in tqdm(range(T), desc="生成SMPL顶点"):
            pose = poses_local[i:i+1].to(device)
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
    
    data = torch.load(pose_file, map_location=device, weights_only=False)
    poses = data['poses']
    
    print(f"帧数: {poses.shape[0]}")
    
    apply_calibration = (motion_id != 'm1')
    if apply_calibration:
        print("应用T-pose校准 (基于m1)")
    else:
        print("T-pose参考数据，不进行校准")
    
    vertices_list = poses_to_smpl_vertices(
        poses, m, device, 
        tpose_ref=tpose_ref, 
        apply_calibration=apply_calibration
    )
    
    output_path = os.path.join(OUTPUT_DIR, f'{motion_id}_animation.mp4')
    print(f"渲染到: {output_path}")
    frames = render_mesh_sequence(vertices_list, faces, output_path, fps=FPS)
    
    print(f"✓ 完成: {output_path} ({len(frames)}帧)")
    return output_path

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("SMPL人体模型渲染 - Linux服务器版本")
    print("="*60)
    print(f"设备: {device}")
    print(f"动作列表: {MOTION_IDS}")
    print(f"m1 = T-pose校准, m2-m7 = 动作序列")
    print("="*60)
    
    print("\n加载SMPL模型...")
    m = kinematic_model.ParametricModel(SMPL_FILE, device=device)
    faces = m.face
    print(f"SMPL模型已加载 (面数: {len(faces)})")
    
    print("\n加载T-pose校准数据...")
    tpose_ref = load_tpose_calibration(device)
    
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
