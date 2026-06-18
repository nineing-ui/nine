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

# ===== 计算单个 Top-K 的任务平均重叠率 =====
def iou_pairwise(mask_array: np.ndarray) -> np.ndarray:
    """
    mask_array: (num_tasks, D) 的二值掩码（0/1）
    return: (num_tasks, num_tasks) 的 IoU 矩阵
    """
    bin_mat = (mask_array > 0).astype(np.uint8)
    inter = bin_mat @ bin_mat.T  # 交集
    sums = bin_mat.sum(axis=1)
    union = sums[:, None] + sums[None, :] - inter
    union = np.maximum(union, 1e-10)  # 防零
    return inter / union



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


masks_list = []
top = 21
avg_overlap = 0.0

for id in range(4): #层数
    for i in range(10):
        e_k = task_encoder.encode(seq_tasks[i]["hint"])[np.newaxis, :] 
        mask_l_t = mask_interface.get_mask(layer_id=id, task_embedding=e_k[0],topk=top) # 得到第 i 层第t个任务的初始掩码
        masks_list.append(mask_l_t)
    mask_array = np.array(masks_list)
    print(f"对num_layers_{id}中组合后的输出形状{mask_array.shape}")
    # 计算 IoU 矩阵与平均重叠率
    iou_mat = iou_pairwise(mask_array)
    n = iou_mat.shape[0]
    iu, ju = np.triu_indices(n, k=1)      # 只取 i<j
    overlap = float(iou_mat[iu, ju].mean())
    avg_overlap += overlap

    # 可选：记录当前 Top-K（若你通过外部配置或 MaskInterface 设置）
    CURRENT_TOPK = top  # 比如 11/15/19；若没有可留空或写成 int
    print(f"[Layer {id}] Top-K={CURRENT_TOPK} → Average Task Overlap (IoU) = {overlap:.4f}")

avg_overlap = avg_overlap/4
print(f"Top-K={CURRENT_TOPK} → Average Task Overlap (IoU) = {avg_overlap:.4f}")




