import os
import numpy as np
from typing import Tuple, Dict

from som_mask.som_network import SOMNetwork


class MaskInterface:
    """Interface for loading layer-wise pretrained SOMs and generating masks.

    The pretrained SOMs are expected to be produced from benchmark-provided
    natural-language task descriptions. At training/evaluation time, the current
    task index selects the corresponding precomputed semantic mask. No trajectory,
    reward, replay-buffer, expert-demonstration, or policy-gradient information is
    used by the SOM allocator itself.
    """

    def __init__(
        self,
        input_dim: int = 384,
        num_layers: int = 4,
        som_size: Tuple[int, int] = (8, 8),
        mask_dim: int = 1024,
        sparsity: float = 0.3,
        weight_dir: str = "trained_soms",
        default_topk: int = 19,
    ):
        """Load one pretrained SOM per policy layer.

        Args:
            input_dim: Dimension of the task-description embedding.
            num_layers: Number of layer-wise SOM allocators.
            som_size: SOM grid size. The default matches ``train_som.py``.
            mask_dim: Width of the generated neuron-level mask.
            sparsity: Retained for compatibility with older mask-template code.
            weight_dir: Directory containing ``som_layer_{layer_id}.npy`` files.
            default_topk: Default number of SOM units activated per layer.
        """
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.som_size = som_size
        self.mask_dim = mask_dim
        self.sparsity = sparsity
        self.weight_dir = weight_dir
        self.default_topk = default_topk
        self.som_networks: Dict[int, SOMNetwork] = {}

        for layer_id in range(num_layers):
            som = SOMNetwork(
                input_dim=input_dim,
                som_size=som_size,
                random_seed=100 + layer_id,
            )
            weight_path = os.path.join(weight_dir, f"som_layer_{layer_id}.npy")
            som.load_weights(weight_path)
            self.som_networks[layer_id] = som

    def get_mask(self, layer_id: int, task_embedding: np.ndarray, topk: int) -> np.ndarray:
        """Generate the neuron-level mask for one layer from a task embedding."""
        if layer_id not in self.som_networks:
            raise KeyError(f"Unknown layer_id={layer_id}. Available layers: {list(self.som_networks)}")
        return self.som_networks[layer_id].get_mask(
            task_embedding,
            k=topk,
            mask_dim=self.mask_dim,
        )

    def get_mask_with_coord(
        self,
        layer_id: int,
        task_embedding: np.ndarray,
        topk: int = None,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Return the generated mask and the best-matching SOM coordinate."""
        if topk is None:
            topk = self.default_topk
        if layer_id not in self.som_networks:
            raise KeyError(f"Unknown layer_id={layer_id}. Available layers: {list(self.som_networks)}")
        som = self.som_networks[layer_id]
        return som.get_mask(task_embedding, k=topk, mask_dim=self.mask_dim), som.get_bmu(task_embedding)
