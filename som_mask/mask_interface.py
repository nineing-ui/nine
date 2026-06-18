import os
import numpy as np
from typing import Tuple, Dict


from som_mask.som_mask_selector import SOMMaskSelector
from som_mask.som_network import SOMNetwork
from som_mask.mask_template_generator import generate_all_layer_templates

class MaskInterface:
    def __init__(
        self,
        input_dim: int = 1024,
        num_layers: int = 4,
        som_size: Tuple[int, int] = (10, 10),
        mask_dim: int = 1024,
        sparsity: float = 0.3,
        weight_dir: str = "trained_soms"  # <== 加载预训练模型路径
    ):
        """
        初始化 SOM 掩码生成器，加载预训练权重
        """
        self.num_layers = num_layers
        self.mask_dim = mask_dim
        self.som_networks = {}

        # Step 1: 加载每层的 SOM 网络（已训练好的 .npy 权重）
        for layer_id in range(num_layers):
            som = SOMNetwork(input_dim=input_dim, som_size=som_size,random_seed = 100+layer_id)
            weight_path = os.path.join(weight_dir, f"som_layer_{layer_id}.npy")
            som.load_weights(weight_path)
            self.som_networks[layer_id] = som

        # Step 2: 掩码模板生成（或从本地加载）
        

        # Step 3: 初始化选择器
        

    
    def get_mask(self, layer_id: int, task_embedding: np.ndarray,topk:int) -> np.ndarray:
        return self.som_networks[layer_id].get_mask(task_embedding,k=topk,mask_dim=self.mask_dim)


    def get_mask_with_coord(self, layer_id: int, task_embedding: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        return self.selector.get_mask_with_coord(layer_id, task_embedding)
