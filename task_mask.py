from copy import deepcopy
import functools
from typing import Optional, Tuple, Any
import numpy as np
from numpy import linalg, subtract
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块显卡
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from jax import random
import jax
import jax.numpy as jnp
import numpy as np
from optax import global_norm
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from flax import linen as nn
from flax.core import freeze, unfreeze, FrozenDict
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import sparse_encode

import jaxrl.networks.common as utils_fn
from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, TrainState, PRNGKey, Params, \
    MPNTrainState
from jaxrl.dict_learning.task_dict import OnlineDictLearnerV2
from jaxrl.networks.common import  PRNGKey, default_init
from som_mask.mask_interface import MaskInterface #接入SOM
from sentence_transformers import SentenceTransformer
from continual_world import TASK_SEQS
def jaccard_similarity(matrix):
    """计算行间 Jaccard 相似度矩阵"""
    # 转换二进制值 (确保是整数 0/1)
    bin_matrix = matrix.astype(bool).astype(int)
    
    # 计算交集和并集
    dot_product = np.dot(bin_matrix, bin_matrix.T)
    union = (bin_matrix.sum(axis=1)[:, None] + bin_matrix.sum(axis=1) - dot_product)
    
    # 避免除以 0
    union = np.maximum(union, 1e-10)  # 更安全的处理方式
    jaccard = dot_product / union
    return jaccard

def create_heatmap(input_csv, output_csv=None, output_png=None,layer_id=0):
    """
    从CSV读取数据，计算行相似度，保存结果并生成热力图
    
    参数:
    input_csv: 输入的CSV文件路径
    output_csv: 相似度矩阵保存路径（可选）
    output_png: 热力图保存路径（可选）
    """
    # 步骤1: 从CSV读取数据
    print(f"正在读取数据: {input_csv}")
    df = pd.read_csv(input_csv)
    data = df.values.astype(float)
    print(f"数据形状: {data.shape} (行×列)")
    
    # 步骤2: 计算Jaccard相似度矩阵
    print("计算行相似度...")
    sim_matrix = jaccard_similarity(data)
    
    # 步骤3: 保存相似度矩阵到CSV（可选）
    if output_csv:
        sim_df = pd.DataFrame(sim_matrix, index=df.index, columns=df.index)
        sim_df.to_csv(output_csv)
        print(f"相似度矩阵已保存至: {output_csv}")
    
    # 步骤4: 绘制热力图
    plt.figure(figsize=(10, 8))#12,10
    ax = sns.heatmap(
        sim_matrix,
        cmap='Blues',
        vmin=0,
        vmax=1,
        annot=True,
        fmt='.2f',
        square=True,
        xticklabels = [f"task{i}" for i in range(1, 11)],
        yticklabels = [f"task{i}" for i in range(1, 11)],
        cbar_kws={"label": f"layer_{layer_id} Similarity"},#更改数字
        annot_kws={  # 关键：设置注释样式
        # "weight": "bold",  # 加粗（可选值：'normal', 'bold', 'heavy', 'light', 'ultrabold', 'ultralight'）
        "size": 12,       # 可选：调整字号（默认可能过小，建议根据图表大小调整）
        # "family": "Times New Roman"  # 可选：指定字体（如Times New Roman需确保环境已安装）
        },
        
    )
    cbar = ax.collections[0].colorbar  # 或 ax.figure.colorbar(ax.collections[0])
    # 2. 调整色条刻度数字的字体大小（如12）
    # cbar.ax.tick_params(labelsize=12, weight='bold')  # 数字大小+加粗
    # 3. 调整色条标签"layer_2 Similarity"的字体（大小+加粗）
    cbar.set_label(
        f"layer_{layer_id} Similarity",
        fontsize=20,  # 标签字体大小
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',fontsize=16) # rotation=45表示向左倾斜45度
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right',fontsize=16)
    plt.title(" ")
    plt.xlabel(" ")
    plt.ylabel(" ")
    
    # 保存或显示图像
    if output_png:
        plt.savefig(output_png, bbox_inches='tight', dpi=150)
        print(f"热力图已保存至: {output_png}")
    else:
        plt.show()
    
    return sim_matrix

###start:外部生成子任务掩码###
num_layers = 4
num_tasks = 10
# 模拟生成任务嵌入：每层 20 个任务 × 384 维嵌入
task_name = 'cw10'
seq_tasks = TASK_SEQS[task_name]
task_encoder = SentenceTransformer('all-MiniLM-L12-v2')
#获取输入向量的维度
task_e = task_encoder.encode(seq_tasks[0]["hint"])[np.newaxis, :]
ek_dim = task_e[0].shape[0]


# 生成任务嵌入向量
mask_dict = {}
mask_interface = MaskInterface(
    input_dim=ek_dim,       # 一般为 384 或 1024
    num_layers=4,
    weight_dir="trained_soms"       # 你的权重路径
)

id = 0 # 层数
masks_list = []

for i in range(10):
    e_k = task_encoder.encode(seq_tasks[i]["hint"])[np.newaxis, :] 
    mask_l_t = mask_interface.get_mask(layer_id=id, task_embedding=e_k[0],topk=19) # 得到第 i 层第t个任务的初始掩码
    masks_list.append(mask_l_t)
mask_array = np.array(masks_list)
print(f"对num_layers_{id}中组合后的输出形状{mask_array.shape}")


###生成相似度热力图###
input_file = f"example_layer{id}.csv"
output_sim = f"similarity_matrix_top15.csv"
output_img = f"similarity_heatmap_top15.png"
pd.DataFrame(mask_array).to_csv(input_file, index=False)


# 执行处理流程
sim_matrix = create_heatmap(
    input_csv=input_file,
    output_csv=output_sim,
    output_png=output_img,
    layer_id=id
)

print("处理完成! 结果:")
print(f"· 相似度矩阵: {output_sim}")
print(f"· 热力图: {output_img}")