import numpy as np
from minisom import MiniSom
from typing import Tuple


class SOMNetwork:
    """Layer-wise SOM allocator for semantic subnetwork masks.

    The SOM is trained only on task-description embeddings. It does not consume
    trajectories, rewards, replay buffers, expert demonstrations, policy weights,
    or any environment-interaction data.

    Mask generation is deterministic after SOM pretraining:
      1. compute the distance map between a task embedding and all SOM units;
      2. activate the Top-K closest SOM units;
      3. flatten the binary SOM-unit activation map;
      4. repeat/trim it to the target hidden-layer width.

    This implementation therefore uses a fixed Top-K-and-tiling mapping from SOM
    units to neurons, not a learned propagation matrix.
    """

    def __init__(
        self,
        input_dim: int,
        som_size: Tuple[int, int] = (8, 8),
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        random_seed: int = 42,
    ):
        self.som_x, self.som_y = som_size
        self.input_dim = input_dim
        self.som = MiniSom(
            x=self.som_x,
            y=self.som_y,
            input_len=input_dim,
            sigma=sigma,
            learning_rate=learning_rate,
            random_seed=random_seed,
        )
        self._is_trained = False

    @property
    def num_units(self) -> int:
        return self.som_x * self.som_y

    def train(self, data: np.ndarray, num_iterations: int = 100):
        """Train the SOM from semantic task-description embeddings."""
        self.som.random_weights_init(data)
        self.som.train_random(data, num_iterations)
        self._is_trained = True
        print(f"✅ SOM trained on {len(data)} text-description embeddings for {num_iterations} iterations.")

    def get_bmu(self, e_k: np.ndarray) -> Tuple[int, int]:
        if not self._is_trained:
            raise ValueError("SOM must be trained before calling get_bmu.")
        return self.som.winner(e_k)

    def get_topk_unit_mask(self, e_k: np.ndarray, k: int) -> np.ndarray:
        """Return a flattened binary mask over SOM units.

        Args:
            e_k: Task-description embedding.
            k: Number of nearest SOM units to activate.

        Returns:
            A vector of shape ``(som_x * som_y,)`` with Top-K units set to 1.
        """
        if not self._is_trained:
            raise ValueError("SOM must be trained before calling get_topk_unit_mask.")
        if k <= 0:
            raise ValueError("top-k must be positive.")
        if k > self.num_units:
            raise ValueError(f"top-k={k} exceeds the number of SOM units ({self.num_units}).")

        distances = self.som.activate(e_k)
        top_k_indices = np.argsort(distances, axis=None)[:k]
        unit_mask = np.zeros(self.num_units, dtype=np.float32)
        unit_mask[top_k_indices] = 1.0
        return unit_mask

    def unit_mask_to_neuron_mask(self, unit_mask: np.ndarray, mask_dim: int) -> np.ndarray:
        """Map flattened SOM-unit activations to a neuron-level mask.

        The mapping is fixed and deterministic: the SOM-unit mask is repeated and
        then trimmed to ``mask_dim``. This avoids an implicit or underspecified
        learned/random propagation matrix.
        """
        if mask_dim <= 0:
            raise ValueError("mask_dim must be positive.")
        repeats = int(np.ceil(mask_dim / unit_mask.shape[0]))
        return np.tile(unit_mask, repeats)[:mask_dim].astype(np.float32)

    def get_mask(self, e_k: np.ndarray, k: int, mask_dim: int):
        unit_mask = self.get_topk_unit_mask(e_k, k)
        return self.unit_mask_to_neuron_mask(unit_mask, mask_dim)

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
