from copy import deepcopy
import functools
from typing import Optional, Tuple, Any

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

###start:外部生成子任务掩码###
num_layers = 4
num_tasks = 20
# 模拟生成任务嵌入：每层 20 个任务 × 384 维嵌入
task_name = 'cw20'
seq_tasks = TASK_SEQS[task_name]
task_encoder = SentenceTransformer('all-MiniLM-L12-v2')
task_e = task_encoder.encode(seq_tasks[0]["hint"])[np.newaxis, :]
#获取输入向量的维度
ek_dim = task_e[0].shape[0]


# 生成任务嵌入向量
mask_dict = {}
mask_interface = MaskInterface(
    input_dim=ek_dim,       # 一般为 384 或 1024
    num_layers=4,
    weight_dir="trained_soms"       # 你的权重路径
)
for l in range(num_layers) :
    mask_l_t_list = []
    for i in range(num_tasks):
        e_k = task_encoder.encode(seq_tasks[i]["hint"])[np.newaxis, :] 
        mask_l_t = mask_interface.get_mask(layer_id=l, task_embedding=e_k[0]) # 得到第 i 层第t个任务的初始掩码
        mask_l_t_list.append(mask_l_t)
    combined_output = jnp.stack(mask_l_t_list, axis=0).squeeze()
    print(f"对num_layers_{l}中组合后的输出形状{combined_output.shape}")
    mask_dict[f'layer_{l}'] = combined_output 
###end:外部生成子任务掩码###