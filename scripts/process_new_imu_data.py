"""
处理新的IMU数据
配置：
- 所有IMU坐标系统一：X向上，Y向左，右手系（观察者视角）
- IMU位置：0=头部（坏了，用3号数据），1=左手，2=右手，3=根节点，4=左腿，5=右腿
- m1是T-pose校准
- 目标坐标系：X向左，Y向上，Z向前
"""
import os
import numpy as np
import pandas as pd
import re

def parse_raw_imu_file(filepath):
    """解析原始IMU数据文件"""
    data_records = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # 匹配数据行
        match = re.match(r'\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\|\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
        if match:
            timestamp = float(match.group(1))
            imu_id = int(match.group(2))
            accel_x = float(match.group(3))
            accel_y = float(match.group(4))
            accel_z = float(match.group(5))
            roll = float(match.group(6))
            pitch = float(match.group(7))
            yaw = float(match.group(8))
            
            data_records.append({
                'timestamp': timestamp,
                'imu_id': imu_id,
                'accel_x': accel_x,
                'accel_y': accel_y,
                'accel_z': accel_z,
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw
            })
    
    return data_records

def euler_to_rotation_matrix(roll, pitch, yaw):
    """将欧拉角（度）转换为旋转矩阵"""
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    
    R_y = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    return R_z @ R_y @ R_x

def rotation_matrix_to_euler(R):
    """将旋转矩阵转换为欧拉角（度）"""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def get_coordinate_transform():
    """
    获取坐标变换矩阵
    
    所有IMU统一坐标系：X向上，Y向左，Z向前（右手系）
    目标坐标系：X向左，Y向上，Z向前
    
    变换：
    X_new = Y_old (向左)
    Y_new = X_old (向上)
    Z_new = Z_old (向前)
    """
    transform = np.array([
        [0, 1, 0],   # X_new = Y_old (向左)
        [1, 0, 0],   # Y_new = X_old (向上)
        [0, 0, 1]    # Z_new = Z_old (向前)
    ])
    
    return transform

def apply_coordinate_transform(accel, rotation_matrix):
    """对加速度和旋转数据应用坐标变换"""
    transform = get_coordinate_transform()
    
    # 变换加速度向量
    accel_transformed = transform @ accel
    
    # 变换旋转矩阵: R_new = T @ R_old @ T^T
    rotation_transformed = transform @ rotation_matrix @ transform.T
    
    return accel_transformed, rotation_transformed

def convert_to_csv(input_file, output_file):
    """将TXT文件转换为CSV格式"""
    print(f"转换 {input_file} 到 CSV...")
    
    data_records = parse_raw_imu_file(input_file)
    df = pd.DataFrame(data_records)
    
    # 保存到CSV
    df.to_csv(output_file, index=False)
    print(f"  保存 {len(df)} 条记录到 {output_file}")
    print(f"  时间范围: {df['timestamp'].min():.3f}s - {df['timestamp'].max():.3f}s")
    print(f"  IMU IDs: {sorted(df['imu_id'].unique())}")
    
    return df

def fill_missing_imu0_and_transform(csv_file, output_file):
    """
    填充缺失的IMU 0数据（用IMU 3的数据），然后应用坐标变换
    """
    print(f"\n处理 {csv_file}...")
    
    # 读取原始CSV
    df = pd.read_csv(csv_file)
    
    # 步骤1: 填充IMU 0的数据（用IMU 3的数据）
    filled_data = []
    
    for timestamp in df['timestamp'].unique():
        frame_data = df[df['timestamp'] == timestamp]
        
        # 获取现有IMU数据
        imu_data = {}
        for _, row in frame_data.iterrows():
            imu_data[int(row['imu_id'])] = row
        
        # 如果缺失ID 0，用ID 3的数据填充
        if 0 not in imu_data and 3 in imu_data:
            id0_raw = imu_data[3].copy()
            id0_raw['imu_id'] = 0
            filled_data.append(id0_raw)
        
        # 保留所有现有数据
        for imu_id in sorted(imu_data.keys()):
            filled_data.append(imu_data[imu_id])
    
    df_filled = pd.DataFrame(filled_data).sort_values(['timestamp', 'imu_id']).reset_index(drop=True)
    
    print(f"  填充后的IMU IDs: {sorted(df_filled['imu_id'].unique())}")
    
    # 步骤2: 应用坐标变换
    transformed_data = []
    
    for idx, row in df_filled.iterrows():
        imu_id = int(row['imu_id'])
        accel = np.array([row['accel_x'], row['accel_y'], row['accel_z']])
        rotation_matrix = euler_to_rotation_matrix(row['roll'], row['pitch'], row['yaw'])
        
        # 应用坐标变换
        accel_transformed, rotation_transformed = apply_coordinate_transform(
            accel, rotation_matrix
        )
        
        # 转换回欧拉角
        roll_new, pitch_new, yaw_new = rotation_matrix_to_euler(rotation_transformed)
        
        transformed_data.append({
            'timestamp': row['timestamp'],
            'imu_id': imu_id,
            'accel_x': accel_transformed[0],
            'accel_y': accel_transformed[1],
            'accel_z': accel_transformed[2],
            'roll': roll_new,
            'pitch': pitch_new,
            'yaw': yaw_new
        })
    
    df_out = pd.DataFrame(transformed_data)
    df_out.to_csv(output_file, index=False)
    print(f"  已保存 {len(df_out)} 条记录到 {output_file}")
    
    # 验证重力方向
    avg_accel = df_out[['accel_x', 'accel_y', 'accel_z']].mean()
    print(f"  平均加速度: X={avg_accel['accel_x']:.3f}g, Y={avg_accel['accel_y']:.3f}g, Z={avg_accel['accel_z']:.3f}g")
    if 0.8 < avg_accel['accel_y'] < 1.2:
        print(f"  ✓ Y轴重力正确 (~1.0g)")
    else:
        print(f"  ⚠ 警告: Y轴重力 = {avg_accel['accel_y']:.3f}g")
    
    return df_out

def main():
    txt_dir = './mydata'
    csv_dir = './mydata/csv'
    output_dir = './mydata/processed_final'
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("新IMU数据处理流程")
    print("="*70)
    print("\n配置信息:")
    print("  - 所有IMU统一坐标系: X向上, Y向左, Z向前 (右手系)")
    print("  - IMU位置: 0=头部(坏), 1=左手, 2=右手, 3=根节点, 4=左腿, 5=右腿")
    print("  - 0号IMU数据用3号填充")
    print("  - 目标坐标系: X向左, Y向上, Z向前")
    print("="*70)
    
    # 处理m1-m7文件
    files_to_process = [f'm{i}' for i in range(1, 8)]
    
    for file_name in files_to_process:
        txt_file = os.path.join(txt_dir, f'{file_name}.txt')
        csv_file = os.path.join(csv_dir, f'{file_name}.csv')
        output_file = os.path.join(output_dir, f'{file_name}_final.csv')
        
        if os.path.exists(txt_file):
            print(f"\n[{file_name}] 开始处理...")
            
            # 步骤1: TXT -> CSV
            df_csv = convert_to_csv(txt_file, csv_file)
            
            # 步骤2: 填充IMU 0 + 坐标变换
            df_final = fill_missing_imu0_and_transform(csv_file, output_file)
            
        else:
            print(f"警告: {txt_file} 不存在，跳过...")
    
    print("\n" + "="*70)
    print("处理完成!")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  CSV文件: {csv_dir}/")
    print(f"  处理后文件: {output_dir}/")
    print("\n下一步: 运行 infer_from_csv.py 进行动作重建")

if __name__ == "__main__":
    main()
