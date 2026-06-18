import numpy as np
import jax.numpy as jnp
from minisom import MiniSom
from typing import Tuple

class SOMNetwork:
    def __init__(
        self,
        input_dim: int,
        som_size: Tuple[int, int] = (32, 32),
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        random_seed: int = 42
    ):
        self.som_x, self.som_y = som_size
        self.input_dim = input_dim
        self.som = MiniSom(
            x=self.som_x,
            y=self.som_y,
            input_len=input_dim,
            sigma=sigma,
            learning_rate=learning_rate,
            random_seed=random_seed
        )
        self._is_trained = False
    

    def train(self, data: np.ndarray, num_iterations: int = 100):
        self.som.random_weights_init(data)
        self.som.train_random(data, num_iterations)
        self._is_trained = True
        print(f"✅ SOM trained on {len(data)} samples for {num_iterations} iterations.")

    def get_bmu(self, e_k: np.ndarray) -> Tuple[int, int]:
        if not self._is_trained:
            raise ValueError("SOM must be trained before calling get_bmu.")
        return self.som.winner(e_k)
    
    
    def get_mask(self,e_k: np.ndarray,k:int,mask_dim:int):
        #获取e_k和权重的欧几里得距离
        distances = self.som.activate(e_k)
        #获取距离矩阵的排序下标并拉直距离矩阵
        sortDistance = distances.argsort(axis=None) #直接把som竞争层 folteen拉直
        #获取前k个小距离下标
        top_k_index = sortDistance[:k]
        #转换得到掩码向量应该和隐藏层一样大
        mask = np.zeros_like(sortDistance,dtype='float')
        # print(f"get_mask:mask.shape:{mask.shape}")
        for index in top_k_index:
            mask[index] = 1
        rep = 0
        old_mask_dim = mask.shape[0]#(一定要old_mask_dim小于mask——dim才行)
        while old_mask_dim * rep != mask_dim:
            rep += 1
        mask = np.tile(mask, rep)
        # print(f"get_mask:new_mask.shape:{mask.shape}")
        return mask

        

    def reinforce(self, e_k: np.ndarray, eta: float = 0.1):
        num_iterations = 100
        if not self._is_trained:
            raise ValueError("SOM must be trained before reinforcement.")
        self.som.update(e_k, self.som.winner(e_k), eta, num_iterations)

    def save_weights(self, path: str):
        np.save(path, self.som.get_weights())
        print(f"✅ SOM weights saved to {path}")

    def load_weights(self, path: str):
        weights = np.load(path)
        self.som._weights = weights
        self._is_trained = True
        print(f"✅ SOM weights loaded from {path}")
