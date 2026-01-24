"""
Extract IMU data from ODT files and convert to motion reconstruction format
Similar to the mydata/*.txt format used in this project
"""
import os
import re
from odf import text, teletype
from odf.opendocument import load

def extract_text_from_odt(odt_file):
    """Extract all text content from an ODT file"""
    doc = load(odt_file)
    all_text = []
    
    # Get all paragraphs
    paragraphs = doc.getElementsByType(text.P)
    for para in paragraphs:
        para_text = teletype.extractText(para)
        if para_text.strip():
            all_text.append(para_text.strip())
    
    return all_text

def parse_imu_data_from_text(text_lines):
    """
    Parse IMU data from text lines.
    Expected format similar to mydata/1.txt:
    Rel_Time(s)  | ID |  Accel_X  Accel_Y  Accel_Z |    Roll   Pitch     Yaw
       0.000 | 0  |    0.875   -0.059   -0.524 |  -174.5   -58.6   -28.5
    """
    data_records = []
    
    for line in text_lines:
        # Try to match data lines with pattern: time | ID | accel_x accel_y accel_z | roll pitch yaw
        # More flexible pattern to handle various spacing
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

def write_txt_format(data_records, output_file):
    """
    Write data in the same format as mydata/*.txt files
    """
    with open(output_file, 'w') as f:
        # Write header
        f.write("\n")
        f.write("IMU Data Stream (50Hz)\n")
        f.write("\n")
        f.write("----------------------------------------------------------------------------------------\n")
        f.write("Rel_Time(s)  | ID |  Accel_X  Accel_Y  Accel_Z |    Roll   Pitch     Yaw\n")
        f.write("----------------------------------------------------------------------------------------\n")
        
        # Write data
        for record in data_records:
            f.write(f"   {record['timestamp']:6.3f} | {record['imu_id']}  | "
                   f"{record['accel_x']:8.3f} {record['accel_y']:8.3f} {record['accel_z']:8.3f} | "
                   f"{record['roll']:8.1f} {record['pitch']:8.1f} {record['yaw']:8.1f}\n")

def process_odt_file(odt_file, output_file):
    """Process a single ODT file and convert to TXT format"""
    print(f"\nProcessing {odt_file}...")
    
    # Extract text from ODT
    text_lines = extract_text_from_odt(odt_file)
    print(f"  Extracted {len(text_lines)} lines of text")
    
    # Parse IMU data
    data_records = parse_imu_data_from_text(text_lines)
    print(f"  Parsed {len(data_records)} data records")
    
    if len(data_records) == 0:
        print(f"  Warning: No data records found!")
        print(f"  First few lines of text:")
        for i, line in enumerate(text_lines[:10]):
            print(f"    {i}: {line}")
        return False
    
    # Get time range and IMU IDs
    timestamps = [r['timestamp'] for r in data_records]
    imu_ids = sorted(set(r['imu_id'] for r in data_records))
    
    print(f"  Time range: {min(timestamps):.3f}s - {max(timestamps):.3f}s")
    print(f"  Duration: {max(timestamps) - min(timestamps):.3f}s")
    print(f"  IMU IDs: {imu_ids}")
    
    # Write to TXT file
    write_txt_format(data_records, output_file)
    print(f"  Saved to {output_file}")
    
    return True

def main():
    data_dir = './data'
    output_dir = './mydata'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("ODT to TXT Converter for Motion Reconstruction")
    print("="*70)
    
    # Process all ODT files (m1.odt through m7.odt)
    odt_files = [f'm{i}.odt' for i in range(1, 8)]
    
    success_count = 0
    for odt_file in odt_files:
        input_path = os.path.join(data_dir, odt_file)
        output_path = os.path.join(output_dir, odt_file.replace('.odt', '.txt'))
        
        if os.path.exists(input_path):
            if process_odt_file(input_path, output_path):
                success_count += 1
        else:
            print(f"Warning: {input_path} not found, skipping...")
    
    print("\n" + "="*70)
    print(f"Conversion completed! {success_count}/{len(odt_files)} files processed successfully")
    print("="*70)
    print(f"\nOutput files saved in: {output_dir}")
    print("\nNext steps:")
    print("1. Run convert_raw_to_csv.py to convert TXT to CSV")
    print("2. Run process_imu_correct.py to process and transform coordinates")
    print("3. Run infer_from_csv.py for motion reconstruction")

if __name__ == "__main__":
    main()
