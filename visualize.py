"""
FIP推理结果3D可视化脚本
使用trimesh和pyrender渲染人体姿态重建结果动画
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'human_body_prior'))

import torch
import numpy as np
import trimesh
import pyrender
from tqdm import tqdm
import time
import imageio

from src import math
from collections import OrderedDict
from human_body_prior.body_model.body_model import BodyModel
from src import functions as sf
from model.net import FIP
from src.eval_tools import get_global_pose, glb2local
from src import kinematic_model

# ============== 可修改的参数 ==============
SEQ_IDS = [0]      # 要渲染的序列列表，可以是单个如[0]或多个如[0,1,2,3]
MAX_FRAMES = None        # 最大帧数，None=渲染整个序列，设置数字如500只渲染前500帧
FPS = 60                 # 视频帧率
RENDER_COMPARISON = False # 是否渲染对比视频（蓝色预测 vs 绿色真实），False只渲染预测结果
# ==========================================

# 配置
class config:
    dipimu_dir = 'data/imu_test.pt'
    dipimu_betas = 'data/dip_betas.pt'
    support_dir = 'data/support_data'
    smpl_file = 'data/SMPL_male.pkl'
    model_dir = "ckpt/best_model.pt"

body_parm = [[0,186,84.24332236,55.0763866],[0,178, 80.61995365, 52.70750976],
             [0,187, 84.69624344, 55.37249621],[0,170, 76.99658495, 50.33863291],
             [0,180, 81.52579583, 53.29972897],[100,172, 78.59580262, 51.98239681],[0,178, 80.61995365, 52.70750976],
             [0,180, 81.52579583, 53.29972897],[0,187, 84.69624344, 55.37249621],[0,181, 76.99658495, 50.33863291]]
body_parm = torch.tensor(body_parm)/100
Jtr_parent = [-1, -1, -1, 0, 1, 2, 3, 4, 5, 8, 8, 8, 9, 10, 11, 13, 14, 15, 16]
Jtr_id = [1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20,21,0]
rot_id = [1,2,3,4,5,6,9,12,13,14,15,16,17,18,19,0]


def render_mesh_sequence(vertices_list, faces, output_path='output_animation.mp4', fps=30):
    """
    渲染mesh序列为视频
    vertices_list: list of [num_vertices, 3] numpy arrays
    faces: [num_faces, 3] numpy array
    """
    # 设置offscreen渲染
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.3, 0.3, 0.3])
    
    # 相机设置
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # 光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)
    
    # 渲染器
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    
    frames = []
    mesh_node = None
    
    for i, verts in enumerate(tqdm(vertices_list, desc="渲染", leave=False)):
        # 创建mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.visual.vertex_colors = [200, 200, 200, 255]
        
        # 转换为pyrender mesh
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        
        # 更新场景中的mesh
        if mesh_node is not None:
            scene.remove_node(mesh_node)
        mesh_node = scene.add(py_mesh)
        
        # 渲染
        color, _ = r.render(scene)
        frames.append(color)
    
    r.delete()
    
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return frames


def render_comparison(pred_vertices_list, gt_vertices_list, faces, output_path='comparison.mp4', fps=30):
    """
    并排渲染预测和真实mesh
    """
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.3, 0.3, 0.3])
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)
    
    r = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=480)
    
    frames = []
    pred_node = None
    gt_node = None
    
    for i, (pred_verts, gt_verts) in enumerate(tqdm(zip(pred_vertices_list, gt_vertices_list), desc="渲染对比", leave=False, total=len(pred_vertices_list))):
        # 预测mesh (蓝色) - 左边
        pred_verts_shifted = pred_verts.copy()
        pred_verts_shifted[:, 0] -= 0.6
        pred_mesh = trimesh.Trimesh(vertices=pred_verts_shifted, faces=faces)
        pred_mesh.visual.vertex_colors = [100, 150, 255, 255]  # 蓝色
        
        # 真实mesh (绿色) - 右边
        gt_verts_shifted = gt_verts.copy()
        gt_verts_shifted[:, 0] += 0.6
        gt_mesh = trimesh.Trimesh(vertices=gt_verts_shifted, faces=faces)
        gt_mesh.visual.vertex_colors = [100, 255, 150, 255]  # 绿色
        
        # 更新场景
        if pred_node is not None:
            scene.remove_node(pred_node)
        if gt_node is not None:
            scene.remove_node(gt_node)
            
        pred_node = scene.add(pyrender.Mesh.from_trimesh(pred_mesh, smooth=True))
        gt_node = scene.add(pyrender.Mesh.from_trimesh(gt_mesh, smooth=True))
        
        color, _ = r.render(scene)
        frames.append(color)
    
    r.delete()
    
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return frames


def process_sequence(seq_id, model, m, faces, bm, data, meta_data, device):
    """处理单个序列"""
    acc = data['acc'][seq_id]
    ori = data['ori'][seq_id]
    token = data['tokens'][seq_id]
    pose_gt_data = data['pose'][seq_id]
    
    # 计算T-pose
    with torch.no_grad():
        betas = torch.from_numpy(meta_data[f's{token}']).to(device)
        body = bm(betas=betas)
        T_pose_data = body.Jtr[0,:22].cpu().numpy()
    
    T_pose = torch.tensor(T_pose_data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] - T_pose_data[0])
    T_pose = torch.cat([T_pose[:3], T_pose[3:] - T_pose[Jtr_parent[3:]]])
    
    # 预处理IMU数据
    acc = ori[:, 5:].transpose(-1, -2) @ (acc - acc[:, 5:]).unsqueeze(-1)
    acc = acc.squeeze(-1)
    ori[:, :5] = ori[:, 5:].transpose(-1, -2) @ ori[:, :5]
    acc, ori = acc.reshape(-1, 18).to(device) / 30, ori.flatten(1).to(device)
    
    body_p = body_parm[token-1].unsqueeze(0).to(device)
    
    # 获取真实姿态
    pose_t = math.axis_angle_to_rotation_matrix(pose_gt_data).view(-1, 24, 3, 3)
    pose_t_global = get_global_pose(pose_t)[:, rot_id[:-1]]
    pose_ini = pose_t_global[0].unsqueeze(0).to(device)
    
    # 帧数限制
    total_frames = len(acc)
    max_frames = total_frames if MAX_FRAMES is None else min(MAX_FRAMES, total_frames)
    
    # 在线推理
    model.reset(pose_ini, body_p)
    poses_p = []
    integ_hc, hip_hc, spine_hc = None, None, None
    
    with torch.no_grad():
        for i in tqdm(range(max_frames), desc=f"seq{seq_id} 推理", leave=False):
            acc_ = acc[i:i+1].unsqueeze(0)
            ori_ = ori[i:i+1].unsqueeze(0)
            rot, _, _, _, integ_hc, spine_hc, hip_hc = model.forward_online(
                acc_, ori_, integ_hc=integ_hc, hip_hc=hip_hc, spine_hc=spine_hc
            )
            poses_p.append(rot.squeeze())
    
    pose_p = torch.stack(poses_p).reshape(-1, 15, 3, 3).cpu()
    pose_t_global = pose_t_global[:max_frames]
    
    # 转换为local pose
    pose_p_local = glb2local(pose_p)
    pose_t_local = glb2local(pose_t_global)
    
    # 生成mesh顶点
    pred_vertices_list = []
    gt_vertices_list = []
    
    with torch.no_grad():
        for i in tqdm(range(len(pose_p_local)), desc=f"seq{seq_id} mesh", leave=False):
            pose_pred = pose_p_local[i:i+1].to(device)
            pose_pred[:, [0, 7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=device)
            _, _, mesh_p = m.forward_kinematics(pose_pred, calc_mesh=True)
            pred_vertices_list.append(mesh_p[0].cpu().numpy())
            
            if RENDER_COMPARISON:
                pose_gt = pose_t_local[i:i+1].to(device)
                pose_gt[:, [0, 7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=device)
                _, _, mesh_t = m.forward_kinematics(pose_gt, calc_mesh=True)
                gt_vertices_list.append(mesh_t[0].cpu().numpy())
    
    return pred_vertices_list, gt_vertices_list, max_frames


def main():
    path = config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device} | 序列: {SEQ_IDS} | 对比视频: {RENDER_COMPARISON}")
    
    # 加载模型（只加载一次）
    model = FIP()
    model.load_state_dict(torch.load(path.model_dir, weights_only=False))
    model.to(device)
    model.eval()
    
    m = kinematic_model.ParametricModel(path.smpl_file, device=device)
    faces = m.face
    
    data = torch.load(path.dipimu_dir, weights_only=False)
    meta_data = torch.load(path.dipimu_betas, weights_only=False)
    
    bm_fname = os.path.join(path.support_dir, 'body_models/smplh/male/model.npz')
    dmpl_fname = os.path.join(path.support_dir, 'body_models/dmpls/male/model.npz')
    bm = BodyModel(bm_fname=bm_fname, num_betas=16, num_dmpls=8, dmpl_fname=dmpl_fname).to(device)
    
    total_sequences = len(data['acc'])
    
    # 批量处理序列
    for seq_id in SEQ_IDS:
        if seq_id >= total_sequences:
            print(f"跳过序列{seq_id}（超出范围0-{total_sequences-1}）")
            continue
            
        print(f"\n[序列 {seq_id}] 处理中...")
        pred_verts, gt_verts, n_frames = process_sequence(
            seq_id, model, m, faces, bm, data, meta_data, device
        )
        
        # 渲染预测视频
        render_mesh_sequence(pred_verts, faces, f'output_seq{seq_id}_prediction.mp4', fps=FPS)
        
        # 渲染对比视频（可选）
        if RENDER_COMPARISON:
            render_comparison(pred_verts, gt_verts, faces, f'output_seq{seq_id}_comparison.mp4', fps=FPS)
        
        print(f"[序列 {seq_id}] 完成 ({n_frames}帧)")
    
    del model
    torch.cuda.empty_cache()
    print(f"\n全部完成! 共处理 {len(SEQ_IDS)} 个序列")


if __name__ == '__main__':
    main()
