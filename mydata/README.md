# 数据目录

## 结构
```
mydata/
├── m1.txt - m7.txt           # 从ODT提取的原始数据
├── csv/                      # CSV格式数据
│   └── m1.csv - m7.csv
└── processed_final/          # 处理后的数据和结果
    ├── m1_final.csv - m7_final.csv    # 坐标变换后的数据
    └── m1_poses.pt - m7_poses.pt      # 动作重建结果
```

## 生成流程
1. `scripts/extract_odt_data.py` → m*.txt
2. `scripts/process_new_imu_data.py` → csv/ 和 processed_final/*_final.csv
3. `scripts/infer_new_data.py` → processed_final/*_poses.pt
4. `scripts/visualize_new_data.py` → output_m*.mp4
