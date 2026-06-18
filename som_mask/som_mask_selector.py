import numpy as np
from typing import Dict, Tuple

class SOMMaskSelector:
    def __init__(
        self,
        som_networks: Dict[int, object],
        mask_templates: Dict[int, Dict[Tuple[int, int], np.ndarray]]
    ):
        """
        :param som_networks: {layer_id: SOMNetwork}
        :param mask_templates: {layer_id: {(i,j): mask}}
        """
        self.som_networks = som_networks
        self.mask_templates = mask_templates

    def get_mask(self, layer_id: int, task_embedding: np.ndarray) -> np.ndarray:
        som = self.som_networks
        coord = som.get_bmu(task_embedding)
        if coord not in self.mask_templates[layer_id]:
            raise ValueError(f"Invalid SOM coord {coord} for layer {layer_id}")
        return self.mask_templates[layer_id][coord]

    def get_mask_with_coord(self, layer_id: int, task_embedding: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        som = self.som_networks[layer_id]
        coord = som.get_bmu(task_embedding)
        return self.mask_templates[layer_id][coord], coord
