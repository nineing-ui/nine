import sys
sys.path.append('/data/lijie/somAsw')
import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二块显卡
import time
from typing import Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from continual_world import SOM_PRETRAIN_DESCRIPTION_CORPUS
from som_mask import som_network


SOM_PROTOCOL_NOTE = (
    "The SOM pretraining corpus consists only of benchmark-provided natural-language "
    "task descriptions. It is used to construct a semantic topology for subnetwork "
    "allocation and does not include trajectories, rewards, expert demonstrations, "
    "learned policies, gradients, replay buffers, or environment-interaction data."
)


def train_all_layer_soms(
    layer_embeddings: Dict[int, np.ndarray],
    input_dim: int = 384,
    som_size: Tuple[int, int] = (8, 8),
    num_iterations: int = 100,
    output_dir: str = "trained_soms",
):
    """Train and save one SOM allocator for each policy layer.

    Args:
        layer_embeddings: Mapping from layer id to an array with shape
            ``[num_descriptions, input_dim]``.
        input_dim: Dimension of the task-description embeddings.
        som_size: Two-dimensional SOM grid size.
        num_iterations: Number of SOM training iterations.
        output_dir: Directory where layer-wise SOM weights are saved.

    Note:
        The input embeddings should be generated from textual task descriptions
        only. The SOM allocator is intentionally separated from RL interaction data
        to make the information assumption explicit in the code and manuscript.
    """
    os.makedirs(output_dir, exist_ok=True)
    for layer_id, embeds in layer_embeddings.items():
        print(f"🚀 Training SOM for Layer {layer_id} ...")
        som = som_network.SOMNetwork(
            input_dim=input_dim,
            som_size=som_size,
            random_seed=layer_id,
        )
        som.train(embeds, num_iterations=num_iterations)
        weight_path = os.path.join(output_dir, f"som_layer_{layer_id}.npy")
        som.save_weights(weight_path)

    print("🎉 All SOM layers trained and saved.")


def save_som_metadata(
    output_dir: str,
    descriptions,
    encoder_name: str,
    num_layers: int,
    input_dim: int,
    som_size: Tuple[int, int],
    num_iterations: int,
    elapsed_time: float,
):
    """Write a metadata file documenting the SOM pretraining protocol."""
    metadata = {
        "protocol": "task-description-conditioned continual reinforcement learning",
        "som_pretraining_source": "benchmark_provided_text_descriptions",
        "uses_all_benchmark_text_descriptions_before_policy_learning": True,
        "uses_future_environment_interaction_data": False,
        "uses_trajectories": False,
        "uses_rewards": False,
        "uses_expert_demonstrations": False,
        "uses_learned_policies": False,
        "uses_gradients": False,
        "uses_replay_buffers": False,
        "uses_environment_interactions": False,
        "note": SOM_PROTOCOL_NOTE,
        "encoder": encoder_name,
        "num_descriptions": len(descriptions),
        "descriptions": list(descriptions),
        "num_layers": num_layers,
        "input_dim": input_dim,
        "som_size": list(som_size),
        "num_iterations": num_iterations,
        "elapsed_time_seconds": elapsed_time,
    }
    metadata_path = os.path.join(output_dir, "som_pretraining_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"📝 SOM pretraining metadata saved to {metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain layer-wise SOMs from benchmark task descriptions.")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--input_dim", type=int, default=384)
    parser.add_argument("--som_x", type=int, default=8)
    parser.add_argument("--som_y", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="trained_soms")
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L12-v2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    som_size = (args.som_x, args.som_y)

    # Benchmark-level textual description corpus for SOM pretraining.
    # This corpus is used only to build a semantic topology for subnetwork
    # allocation. It does not include trajectories, rewards, policies, gradients,
    # replay buffers, demonstrations, or environment interaction data.
    description_corpus = SOM_PRETRAIN_DESCRIPTION_CORPUS
    num_descriptions = len(description_corpus)

    task_encoder = SentenceTransformer(args.encoder)

    description_embeddings = []
    for description in description_corpus:
        task_e = task_encoder.encode(description)[np.newaxis, :]
        description_embeddings.append(task_e[0])
    description_embeddings = np.array(description_embeddings)

    train_embed = {}
    for layer_id in range(args.num_layers):
        # Each layer receives the same semantic corpus but learns an independent
        # SOM topology under a layer-specific random seed.
        train_embed[layer_id] = description_embeddings.copy()

    start_time = time.time()
    train_all_layer_soms(
        layer_embeddings=train_embed,
        input_dim=args.input_dim,
        som_size=som_size,
        num_iterations=args.num_iterations,
        output_dir=args.output_dir,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    save_som_metadata(
        output_dir=args.output_dir,
        descriptions=description_corpus,
        encoder_name=args.encoder,
        num_layers=args.num_layers,
        input_dim=args.input_dim,
        som_size=som_size,
        num_iterations=args.num_iterations,
        elapsed_time=elapsed_time,
    )

    print(f"SOM pretraining corpus size: {num_descriptions}")
    print(f"运行耗时: {elapsed_time:.6f} 秒")
