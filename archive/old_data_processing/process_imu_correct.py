"""
正确的IMU数据处理与坐标变换
关键思路：先在原始格式下填充缺失数据，再进行坐标变换
"""
import os
import numpy as np
import pandas as pd

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

def get_coordinate_transform(imu_id):
    """
    获取每个IMU传感器的坐标变换矩阵。
    
    IMU方向（以人体为中心）：
    - IMU 0,3,4,5: Z向前, X向上, Y向右
    - IMU 2 (左手): X向前, Y向左, Z向上
    - IMU 1 (右手): X向前, Y向左, Z向上 (与ID 2相同)
    
    目标坐标系: X向左, Y向上, Z向前
    """
    if imu_id in [0, 3, 4, 5]:
        transform = np.array([
            [0, -1, 0],   # X_new = -Y_old (向左 = -向右)
            [1, 0, 0],    # Y_new = X_old (向上)
            [0, 0, 1]     # Z_new = Z_old (向前)
        ])
    elif imu_id == 2:
        # ID 2 (左手): X向前, Y向左, Z向上
        transform = np.array([
            [0, 1, 0],   # X_new = Y_old (向左 = 向左)
            [0, 0, 1],   # Y_new = Z_old (向上)
            [1, 0, 0]    # Z_new = X_old (向前)
        ])
    elif imu_id == 1:
        # ID 1 (右手): X向前, Y向左, Z向上
        transform = np.array([
            [0, 1, 0],   # X_new = Y_old (向左 = 向左)
            [0, 0, 1],   # Y_new = Z_old (向上)
            [1, 0, 0]    # Z_new = X_old (向前)
        ])
    else:
        transform = np.eye(3)
    
    return transform

def apply_coordinate_transform(accel, rotation_matrix, imu_id):
    """对加速度和旋转数据应用坐标变换"""
    transform = get_coordinate_transform(imu_id)
    
    # 变换加速度向量
    accel_transformed = transform @ accel
    
    # 变换旋转矩阵: R_new = T @ R_old @ T^T
    rotation_transformed = transform @ rotation_matrix @ transform.T
    
    return accel_transformed, rotation_transformed

def fill_and_transform_data(csv_file, output_file, id2_avg=None):
    """
    在原始格式下填充缺失的IMU数据，然后应用坐标变换
    """
    print(f"\n正在处理 {csv_file}...")
    
    # 读取原始CSV
    df = pd.read_csv(csv_file)
    
    # 步骤 1: 在原始格式下填充缺失数据
    filled_data = []
    
    for timestamp in df['timestamp'].unique():
        frame_data = df[df['timestamp'] == timestamp]
        
        # 获取现有IMU数据
        imu_data = {}
        for _, row in frame_data.iterrows():
            imu_data[int(row['imu_id'])] = row
        
        # 如果缺失ID 1则填充
        if 1 not in imu_data:
            if 2 in imu_data:
                # 对于 1.csv: 从同文件的ID 2复制
                id1_raw = imu_data[2].copy()
                id1_raw['imu_id'] = 1
                filled_data.append(id1_raw)
            elif id2_avg is not None:
                # 对于 2.csv/3.csv: 使用1.csv的平均值
                id1_raw = id2_avg.copy()
                id1_raw['timestamp'] = timestamp
                id1_raw['imu_id'] = 1
                filled_data.append(id1_raw)
        
        # 从ID 5填充ID 3, 4
        if 5 in imu_data:
            for new_id in [3, 4]:
                if new_id not in imu_data:
                    id_raw = imu_data[5].copy()
                    id_raw['imu_id'] = new_id
                    filled_data.append(id_raw)
        
        # 保留现有数据
        for imu_id in [0, 1, 2, 5]:
            if imu_id in imu_data:
                filled_data.append(imu_data[imu_id])
    
    df_filled = pd.DataFrame(filled_data).sort_values(['timestamp', 'imu_id']).reset_index(drop=True)
    
    # 步骤 2: 应用坐标变换
    transformed_data = []
    
    for idx, row in df_filled.iterrows():
        imu_id = int(row['imu_id'])
        accel = np.array([row['accel_x'], row['accel_y'], row['accel_z']])
        rotation_matrix = euler_to_rotation_matrix(row['roll'], row['pitch'], row['yaw'])
        
        # 应用坐标变换
        accel_transformed, rotation_transformed = apply_coordinate_transform(
            accel, rotation_matrix, imu_id
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
    
    return df_out

def main():
    csv_dir = './mydata/csv'
    output_dir = './mydata/processed_final'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("IMU数据处理 - 正确版本")
    print("="*70)
    print("\n坐标变换:")
    print("  IMU 0,3,4,5: Z向前, X向上, Y向右 → X向左, Y向上, Z向前")
    print("  IMU 2 (左手): X向前, Y向左, Z向上 → X向左, Y向上, Z向前")
    print("  IMU 1 (右手): X向前, Y向左, Z向上 → X向左, Y向上, Z向前")
    print("  注意: ID 1和ID 2的IMU方向相同")
    print("  目标: X向左, Y向上, Z向前")
    print("="*70)
    
    # 先处理 1.csv (T-pose)
    print("\n[1/3] 处理 1.csv (T-pose标定)")
    print("  - 从ID 2原始数据填充ID 1")
    print("  - 从ID 5原始数据填充ID 3,4")
    print("  - 然后应用坐标变换")
    
    df1 = fill_and_transform_data(
        os.path.join(csv_dir, '1.csv'),
        os.path.join(output_dir, '1_final.csv'),
        id2_avg=None
    )
    
    # 计算1.csv中原始ID 2数据的平均值，用于2.csv和3.csv
    df1_raw = pd.read_csv(os.path.join(csv_dir, '1.csv'))
    id2_avg_raw = df1_raw[df1_raw['imu_id'] == 2][['accel_x', 'accel_y', 'accel_z', 'roll', 'pitch', 'yaw']].mean()
    
    print(f"\n1.csv中原始ID 2数据的平均值 (用于填充2.csv/3.csv中的ID 1):")
    print(f"  加速度: ({id2_avg_raw['accel_x']:.3f}, {id2_avg_raw['accel_y']:.3f}, {id2_avg_raw['accel_z']:.3f})")
    print(f"  姿态: Roll={id2_avg_raw['roll']:.1f}°, Pitch={id2_avg_raw['pitch']:.1f}°, Yaw={id2_avg_raw['yaw']:.1f}°")
    
    # 处理 2.csv
    print("\n[2/3] 处理 2.csv")
    print("  - 使用1.csv中原始ID 2的平均值填充ID 1")
    print("  - 从ID 5填充ID 3,4")
    print("  - 然后应用坐标变换")
    
    df2 = fill_and_transform_data(
        os.path.join(csv_dir, '2.csv'),
        os.path.join(output_dir, '2_final.csv'),
        id2_avg=id2_avg_raw
    )
    
    # 处理 3.csv
    print("\n[3/3] 处理 3.csv")
    print("  - 使用1.csv中原始ID 2的平均值填充ID 1")
    print("  - 从ID 5填充ID 3,4")
    print("  - 然后应用坐标变换")
    
    df3 = fill_and_transform_data(
        os.path.join(csv_dir, '3.csv'),
        os.path.join(output_dir, '3_final.csv'),
        id2_avg=id2_avg_raw
    )
    
    print("\n" + "="*70)
    print("处理完成!")
    print("="*70)
    print(f"\n在 {output_dir} 中生成的文件:")
    print("  - 1_final.csv, 2_final.csv, 3_final.csv")
    print("\n验证重力方向 (变换后应在Y轴约为+1.0g)")
    
    for name, df in [('1', df1), ('2', df2), ('3', df3)]:
        avg_accel = df[['accel_x', 'accel_y', 'accel_z']].mean()
        print(f"\n{name}.csv 平均加速度:")
        print(f"  X={avg_accel['accel_x']:.3f}g, Y={avg_accel['accel_y']:.3f}g, Z={avg_accel['accel_z']:.3f}g")
        if 0.8 < avg_accel['accel_y'] < 1.1:
            print(f"  ✓ Y轴重力正确!")
        else:
            print(f"  ✗ 警告: Y轴重力 = {avg_accel['accel_y']:.3f}g")

if __name__ == "__main__":
    main()
