# 新IMU数据处理脚本

## 脚本说明

### 1. extract_odt_data.py
从ODT文件提取IMU数据到TXT格式

### 2. process_new_imu_data.py
- TXT → CSV转换
- 用3号数据填充损坏的0号IMU
- 坐标变换: (X上Y左Z前) → (X左Y上Z前)

### 3. infer_new_data.py
使用FIP模型进行动作重建推理

### 4. visualize_new_data.py
渲染动作重建结果为视频（Mac兼容）

## 按顺序运行
```bash
python extract_odt_data.py
python process_new_imu_data.py
python infer_new_data.py      # 需要模型文件
python visualize_new_data.py  # 需要推理结果
```
