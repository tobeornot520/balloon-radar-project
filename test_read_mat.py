import numpy as np
from scipy.io import loadmat
from pathlib import Path

# 找一个样本文件（用第一个）
mat_files = sorted(Path("data/raw/IQ_Data").glob("*.mat"))
if not mat_files:
    print("❌ 没有找到 .mat 文件")
else:
    sample = mat_files[0]
    print(f"📂 读取文件: {sample.name}")
    
    data = loadmat(sample)
    print(f"🔑 文件中的变量: {list(data.keys())}")
    
    # 检查常见字段名
    for key in ['local_data_H', 'local_data_V', 'H', 'V', 'data']:
        if key in data:
            arr = data[key]
            print(f"   {key}: shape={arr.shape}, dtype={arr.dtype}, is_complex={np.iscomplexobj(arr)}")
