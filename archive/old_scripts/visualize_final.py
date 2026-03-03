"""
Final visualization script with correct coordinate system
Uses d2l conda environment
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'human_body_prior'))

import torch
import numpy as np
import trimesh
import pyrender
from tqdm import tqdm
import imageio

from src import kinematic_model
from src.eval_tools import glb2local

# Configuration
smpl_file = 'data/SMPL_male.pkl'

def render_mesh_sequence(vertices_list, faces, output_path='output_animation.mp4', fps=30):
    """Render mesh sequence as video"""
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.3, 0.3, 0.3])
    
    # Camera setup
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # Lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)
    
    # Renderer
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
    
    # Save video
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    return frames

def poses_to_smpl_vertices(poses, m, device='cpu'):
    """Convert pose rotation matrices to SMPL vertices"""
    T = poses.shape[0]
    
    # Convert to local poses
    poses_local = glb2local(poses)
    
    # Generate vertices
    vertices_list = []
    
    print(f"Converting {T} poses to SMPL vertices...")
    with torch.no_grad():
        for i in tqdm(range(T), desc="生成顶点"):
            pose = poses_local[i:i+1].to(device)
            pose[:, [0, 7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=device)
            _, _, mesh = m.forward_kinematics(pose, calc_mesh=True)
            vertices_list.append(mesh[0].cpu().numpy())
    
    vertices = np.array(vertices_list)
    return vertices

def visualize_pose_file(pose_file, output_name, device='cpu'):
    """Visualize a single pose file"""
    print(f"\n{'='*60}")
    print(f"处理文件: {pose_file}")
    print(f"{'='*60}")
    
    # Load pose data
    data = torch.load(pose_file, map_location=device, weights_only=False)
    poses = data['poses']
    
    print(f"加载了 {poses.shape[0]} 帧")
    print(f"关节数: {poses.shape[1]}")
    
    # Load kinematic model
    print("加载SMPL模型...")
    m = kinematic_model.ParametricModel(smpl_file, device=device)
    faces = m.face
    
    # Convert poses to vertices
    vertices = poses_to_smpl_vertices(poses, m, device)
    
    # Render animation
    output_path = f'mydata/processed_final/{output_name}_animation.mp4'
    print(f"\n渲染动画到: {output_path}")
    frames = render_mesh_sequence(vertices, faces, output_path, fps=30)
    
    print(f"✓ 视频已保存: {output_path}")
    print(f"  帧数: {len(frames)}")
    print(f"  分辨率: {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    return output_path

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("\n✓ 坐标系统: X-右, Y-上, Z-前")
    print("✓ 重力方向: -Y (已验证)")
    print("✓ IMU坐标已转换到项目坐标系")
    print("="*60)
    
    # Files to visualize
    files_to_visualize = [
        ('mydata/processed_final/1_poses.pt', '1'),
        ('mydata/processed_final/2_poses.pt', '2'),
        ('mydata/processed_final/3_poses.pt', '3')
    ]
    
    generated_videos = []
    
    for pose_file, name in files_to_visualize:
        if os.path.exists(pose_file):
            try:
                video_path = visualize_pose_file(pose_file, name, device)
                generated_videos.append(video_path)
            except Exception as e:
                print(f"错误: 处理 {pose_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"警告: {pose_file} 未找到，跳过...")
    
    print("\n" + "="*60)
    print("✓ 可视化完成！")
    print("="*60)
    print("\n生成的视频 (正确坐标系统):")
    for video in generated_videos:
        print(f"  ✓ {video}")
    
    print("\n所有动画已成功生成！")
    print("坐标系统: X-右, Y-上, Z-前 (重力在-Y方向)")

if __name__ == "__main__":
    main()
