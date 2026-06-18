import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取掩码数据（10行×4096列：4层×1024个参数）
mask_df = pd.read_csv("task_masks.csv")
layer_size = 1024  # 每层参数数量
layers = 4         # 隐藏层数量（L0-L3）

# 2. 预计算层间连接参数范围（仅关注L0-L1、L1-L2、L2-L3）
# 每层掩码在CSV中的列范围：L0(0-1023), L1(1024-2047), L2(2048-3071), L3(3072-4095)
layer_ranges = {
    'L0': slice(0, layer_size),
    'L1': slice(layer_size, 2*layer_size),
    'L2': slice(2*layer_size, 3*layer_size),
    'L3': slice(3*layer_size, 4*layer_size)
}

# 3. 初始化累计训练参数标记（三维掩码：层间连接×参数位置）
# 层间连接：L0-L1, L1-L2, L2-L3 → 共3组连接
cumulative_trained = {
    'L0-L1': np.zeros((layer_size, layer_size), dtype=bool),
    'L1-L2': np.zeros((layer_size, layer_size), dtype=bool),
    'L2-L3': np.zeros((layer_size, layer_size), dtype=bool)
}

utilization = []  # 存储每个任务的利用率

for task_idx in range(10):  # 遍历10个任务
    # 获取当前任务各层掩码（转为布尔值）
    mask_L0 = mask_df.iloc[task_idx, layer_ranges['L0']].values.astype(bool)
    mask_L1 = mask_df.iloc[task_idx, layer_ranges['L1']].values.astype(bool)
    mask_L2 = mask_df.iloc[task_idx, layer_ranges['L2']].values.astype(bool)
    mask_L3 = mask_df.iloc[task_idx, layer_ranges['L3']].values.astype(bool)
    
    # 计算当前任务各层间的训练参数（逐连接判断）
    # L0-L1连接：L0掩码为1 且 L1掩码为1
    current_L0L1 = np.outer(mask_L0, mask_L1)  # 外积生成连接掩码
    # L1-L2连接：L1掩码为1 且 L2掩码为1
    current_L1L2 = np.outer(mask_L1, mask_L2)
    # L2-L3连接：L2掩码为1 且 L3掩码为1
    current_L2L3 = np.outer(mask_L2, mask_L3)
    
    # 更新累计训练参数（或运算：保留历史训练参数）
    cumulative_trained['L0-L1'] |= current_L0L1
    cumulative_trained['L1-L2'] |= current_L1L2
    cumulative_trained['L2-L3'] |= current_L2L3
    
    # 计算总训练参数数量
    total_trained = 0
    for conn in cumulative_trained.values():
        total_trained += np.sum(conn)
    
    # 计算利用率（总跨层参数=3×1024²）
    total_params = 3 * (layer_size ** 2)
    utilization.append(total_trained / total_params)
#cotasp数据
cotasp_utilization = [0.03, 0.07, 0.10, 0.13, 0.15, 0.17, 0.20, 0.22, 0.23, 0.24]

# 4. 可视化
task_ids = np.arange(1, 11)  # 任务ID从1到10
plt.plot(task_ids, utilization, marker='o', color='#FF6961',label='Our Method')
# 绘制 CoTASP 的曲线，这里可以设置不同的标记和颜色，比如用 'x' 标记，蓝色线条
plt.plot(task_ids, cotasp_utilization, marker='x', color='blue', label='CoTASP')  
plt.xlabel('Task ID')
plt.ylabel('Network Utilization')
plt.ylim(0, 0.5)  # 匹配示例图范围
plt.legend()
plt.grid(linestyle='--', alpha=0.5)  
plt.savefig('network_utilization.png')
plt.show()
print('Network Utilization:', utilization)