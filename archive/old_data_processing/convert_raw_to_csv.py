"""
Convert raw IMU text data to CSV format
Analyze T-pose data to determine correct coordinate transformations
"""
import re
import numpy as np
import pandas as pd
import os
from collections import defaultdict

def parse_raw_imu_file(filepath):
    """
    Parse raw IMU data file and extract structured data.
    Returns: list of data records
    """
    data_records = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Match data lines with pattern: time | ID | accel_x accel_y accel_z | roll pitch yaw
        match = re.match(r'\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s*\|\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', line)
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

def analyze_tpose_gravity(filepath):
    """
    Analyze T-pose data (1.txt) to determine gravity direction for each IMU.
    In T-pose, acceleration should only be gravity (pointing down).
    """
    print(f"\n{'='*70}")
    print("Analyzing T-pose data to determine coordinate systems...")
    print(f"{'='*70}\n")
    
    data_records = parse_raw_imu_file(filepath)
    df = pd.DataFrame(data_records)
    
    # Group by IMU ID and calculate average acceleration (should be gravity)
    for imu_id in sorted(df['imu_id'].unique()):
        imu_data = df[df['imu_id'] == imu_id]
        avg_accel = np.array([
            imu_data['accel_x'].mean(),
            imu_data['accel_y'].mean(),
            imu_data['accel_z'].mean()
        ])
        magnitude = np.linalg.norm(avg_accel)
        
        print(f"IMU {imu_id}:")
        print(f"  Average Accel: ({avg_accel[0]:.3f}, {avg_accel[1]:.3f}, {avg_accel[2]:.3f})")
        print(f"  Magnitude: {magnitude:.3f}g")
        print(f"  Normalized: ({avg_accel[0]/magnitude:.3f}, {avg_accel[1]/magnitude:.3f}, {avg_accel[2]/magnitude:.3f})")
        
        # Determine which axis is closest to -1g (gravity down)
        abs_values = np.abs(avg_accel)
        dominant_axis = np.argmax(abs_values)
        axis_names = ['X', 'Y', 'Z']
        direction = 'negative' if avg_accel[dominant_axis] < 0 else 'positive'
        
        print(f"  → Gravity direction: {direction} {axis_names[dominant_axis]}-axis")
        print()

def convert_to_csv(input_file, output_file):
    """
    Convert raw IMU text file to CSV format.
    """
    print(f"Converting {input_file} to CSV...")
    
    data_records = parse_raw_imu_file(input_file)
    df = pd.DataFrame(data_records)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"  Saved {len(df)} records to {output_file}")
    print(f"  Time range: {df['timestamp'].min():.3f}s - {df['timestamp'].max():.3f}s")
    print(f"  IMU IDs: {sorted(df['imu_id'].unique())}")
    print()

def main():
    input_dir = './mydata'
    output_dir = './mydata/csv'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("IMU Data Conversion: TXT → CSV")
    print("="*70)
    
    # First, analyze T-pose data to understand coordinate systems
    tpose_file = os.path.join(input_dir, '1.txt')
    if os.path.exists(tpose_file):
        analyze_tpose_gravity(tpose_file)
    
    # Convert all files to CSV
    files_to_convert = ['1.txt', '2.txt', '3.txt']
    
    for filename in files_to_convert:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.replace('.txt', '.csv'))
        
        if os.path.exists(input_file):
            convert_to_csv(input_file, output_file)
        else:
            print(f"Warning: {input_file} not found, skipping...")
    
    print("="*70)
    print("Conversion completed!")
    print(f"CSV files saved in: {output_dir}")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the gravity direction analysis above")
    print("2. Determine correct coordinate transformations")
    print("3. Apply transformations and process data")

if __name__ == "__main__":
    main()
