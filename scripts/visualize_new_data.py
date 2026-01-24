"""
可视化新IMU数据的动作重建结果
支持m1-m7的数据
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

import torch
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')  # Mac上使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import imageio

from src import kinematic_model
from src.eval_tools import glb2local

# 配置
MOTION_IDS = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']  # 要渲染的动作
MAX_FRAMES = None  # None=全部帧，或设置数字限制帧数
FPS = 30
USE_PYRENDER = False  # Mac上可能无法使用pyrender，设为False使用matplotlib

class config:
    smpl_file = 'data/SMPL_male.pkl'
    pose_dir = 'mydata/processed_final'

def render_with_matplotlib(vertices_list, faces, output_path='output.mp4', fps=30):
    """
    使用matplotlib渲染（Mac兼容）
    """
    print(f"使用matplotlib渲染到 {output_path}...")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置视角
    ax.view_init(elev=10, azim=90)
    
    # 计算边界
    all_verts = np.concatenate(vertices_list, axis=0)
    x_range = [all_verts[:, 0].min() - 0.5, all_verts[:, 0].max() + 0.5]
    y_range = [all_verts[:, 1].min() - 0.5, all_verts[:, 1].max() + 0.5]
    z_range = [all_verts[:, 2].min() - 0.5, all_verts[:, 2].max() + 0.5]
    
    frames = []
    
    for i, verts in enumerate(tqdm(vertices_list, desc="渲染帧")):
        ax.clear()
        
        # 绘制mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                       color='lightblue', edgecolor='none', alpha=0.8)
        
        # 设置坐标轴
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {i+1}/{len(vertices_list)}')
        
        # 保存帧
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    plt.close(fig)
    
    # 保存视频
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    print(f"视频已保存: {output_path}")
    return frames

def render_with_pyrender(vertices_list, faces, output_path='output.mp4', fps=30):
    """
    使用pyrender渲染（需要OpenGL支持）
    """
    try:
        import pyrender
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # Mac上尝试使用osmesa
        
        print(f"使用pyrender渲染到 {output_path}...")
        
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.3, 0.3, 0.3])
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 2.5],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)
        
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)
        
        r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        
        frames = []
        mesh_node = None
        
        for i, verts in enumerate(tqdm(vertices_list, desc="渲染帧")):
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.visual.vertex_colors = [200, 200, 200, 255]
            
            py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            
            if mesh_node is not None:
                scene.remove_node(mesh_node)
            mesh_node = scene.add(py_mesh)
            
            color, _ = r.render(scene)
            frames.append(color)
        
        r.delete()
        
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        
        print(f"视频已保存: {output_path}")
        return frames
        
    except Exception as e:
        print(f"pyrender渲染失败: {e}")
        print("回退到matplotlib渲染...")
        return render_with_matplotlib(vertices_list, faces, output_path, fps)

def process_motion(motion_id, m, faces, device):
    """处理单个动作序列"""
    pose_file = os.path.join(config.pose_dir, f'{motion_id}_poses.pt')
    
    if not os.path.exists(pose_file):
        print(f"警告: {pose_file} 不存在，跳过...")
        return None
    
    print(f"\n处理 {motion_id}...")
    
    # 加载姿态数据
    data = torch.load(pose_file, weights_only=False)
    pose_p = data['poses']  # [T, 15, 3, 3]
    
    total_frames = len(pose_p)
    max_frames = total_frames if MAX_FRAMES is None else min(MAX_FRAMES, total_frames)
    pose_p = pose_p[:max_frames]
    
    print(f"  总帧数: {total_frames}, 渲染帧数: {max_frames}")
    
    # 转换为local pose
    pose_p_local = glb2local(pose_p)
    
    # 生成mesh顶点
    vertices_list = []
    
    with torch.no_grad():
        for i in tqdm(range(len(pose_p_local)), desc=f"{motion_id} mesh生成"):
            pose_pred = pose_p_local[i:i+1].to(device)
            # 设置某些关节为单位矩阵
            pose_pred[:, [0, 7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=device)
            _, _, mesh_p = m.forward_kinematics(pose_pred, calc_mesh=True)
            vertices_list.append(mesh_p[0].cpu().numpy())
    
    return vertices_list

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"渲染方式: {'pyrender' if USE_PYRENDER else 'matplotlib'}")
    print(f"动作列表: {MOTION_IDS}")
    
    # 加载SMPL模型
    print("\n加载SMPL模型...")
    m = kinematic_model.ParametricModel(config.smpl_file, device=device)
    faces = m.face
    print(f"SMPL模型加载完成 (顶点数: {m.v_template.shape[0]}, 面数: {len(faces)})")
    
    # 处理每个动作
    for motion_id in MOTION_IDS:
        vertices_list = process_motion(motion_id, m, faces, device)
        
        if vertices_list is None:
            continue
        
        # 渲染视频
        output_path = f'output_{motion_id}.mp4'
        
        if USE_PYRENDER:
            render_with_pyrender(vertices_list, faces, output_path, fps=FPS)
        else:
            render_with_matplotlib(vertices_list, faces, output_path, fps=FPS)
        
        print(f"{motion_id} 完成!")
    
    print("\n全部完成!")

if __name__ == '__main__':
    main()
