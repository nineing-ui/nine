import numpy as np
from typing import Dict, Tuple

def generate_single_mask(dim: int, sparsity: float) -> np.ndarray:
    mask = np.zeros(dim, dtype=np.float32)
    num_active = int(dim * sparsity)
    active_indices = np.random.choice(dim, num_active, replace=False)
    mask[active_indices] = 1.0
    return mask

def mask_overlap_ratio(mask1: np.ndarray, mask2: np.ndarray) -> float:
    return np.sum(mask1 * mask2) / max(np.sum(mask1), 1e-8)

def generate_layer_templates(
    som_size: Tuple[int, int] = (10, 10),
    dim: int = 1024,
    sparsity: float = 0.3,
    overlap_percent: float = 0.8,
    overlap_threshold: float = 0.2,
    random_seed: int = 42
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    为单层 SOM 节点生成稀疏掩码模板，控制重叠结构
    返回结构：{(i, j): mask (1024,)}
    """
    np.random.seed(random_seed)
    total_templates = som_size[0] * som_size[1]
    num_controlled = int(total_templates * overlap_percent)

    base_templates = []
    for _ in range(num_controlled):
        base_templates.append(generate_single_mask(dim, sparsity))

    # 补足其余掩码，确保与 base 的部分重叠
    while len(base_templates) < total_templates:
        ref_mask = base_templates[np.random.randint(num_controlled)]
        new_mask = generate_single_mask(dim, sparsity)

        if mask_overlap_ratio(ref_mask, new_mask) < overlap_threshold:
            ref_indices = np.where(ref_mask == 1.0)[0]
            required_shared = int(dim * sparsity * overlap_threshold)
            shared_indices = np.random.choice(ref_indices, required_shared, replace=False)
            new_mask[shared_indices] = 1.0
        base_templates.append(new_mask)

    # 映射模板到 SOM 坐标
    templates = {}
    idx = 0
    for i in range(som_size[0]):
        for j in range(som_size[1]):
            templates[(i, j)] = base_templates[idx]
            idx += 1

    return templates

def generate_all_layer_templates(
    num_layers: int = 4,
    som_size: Tuple[int, int] = (10, 10),
    dim: int = 1024,
    sparsity: float = 0.3,
    overlap_percent: float = 0.8,
    overlap_threshold: float = 0.2
) -> Dict[int, Dict[Tuple[int, int], np.ndarray]]:
    """
    为所有层生成稀疏掩码模板结构
    返回结构：{layer_id: {(i, j): mask (1024,)}}
    """
    return {
        l: generate_layer_templates(
            som_size=som_size,
            dim=dim,
            sparsity=sparsity,
            overlap_percent=overlap_percent,
            overlap_threshold=overlap_threshold,
            random_seed=42 + l  # 保证每层随机性不同
        )
        for l in range(num_layers)
    }


