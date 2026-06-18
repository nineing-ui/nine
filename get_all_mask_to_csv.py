# 1. 清理冗余导入和设置
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 保留一次即可
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from continual_world import TASK_SEQS
from som_mask.mask_interface import MaskInterface
import time

# 2. 确保目录存在
Path("trained_soms").mkdir(exist_ok=True)

# 3. 重构掩码生成逻辑
task_name = 'cw10'
seq_tasks = TASK_SEQS[task_name]
task_encoder = SentenceTransformer('all-MiniLM-L12-v2')
start_time = time.time() 
mask_interface = MaskInterface(
    input_dim=task_encoder.get_sentence_embedding_dimension(),
    num_layers=4,
    weight_dir="trained_soms"
)

# 4. 高效生成掩码数组
mask_data = []
for i in range(10):
    # 单任务单次编码
    hint = seq_tasks[i]["hint"]
    e_k = task_encoder.encode(hint)  # 直接获取向量
    
    # 生成4层掩码
    masks = [
        mask_interface.get_mask(layer_id=j, task_embedding=e_k)
        for j in range(4)
    ]
    mask_data.append(np.concatenate(masks))  # 展平为1D向量

end_time = time.time()    # 记录结束时间
elapsed_time = end_time - start_time
print(f"运行耗时: {elapsed_time:.6f} 秒")

# 5. 结构化存储为CSV
mask_array = np.vstack(mask_data)  # (10, total_mask_dim)
layer_dims = [1024 for j in range(4)]
col_prefixes = [f"L{j}_" for j in range(4)]

# 生成列名 [L0_0001, L0_0002, ..., L3_1024]
columns = []
for prefix, dim in zip(col_prefixes, layer_dims):
    columns.extend([f"{prefix}{i:04d}" for i in range(dim)])

pd.DataFrame(mask_array, columns=columns).to_csv("task_masks.csv", index=False)