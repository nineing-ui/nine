import sys
sys.path.append('/data/lijie/somAsw')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二块显卡
import numpy as np
from som_mask import som_network

from sentence_transformers import SentenceTransformer
from continual_world import SOM_PRETRAIN_DESCRIPTION_CORPUS
import time


def train_all_layer_soms(
    layer_embeddings: dict,
    input_dim: int = 384,
    som_size: tuple = (8, 8),
    num_iterations: int = 100,
    output_dir: str = "trained_soms"
):
    """
    针对每一层的任务嵌入，训练对应 SOM 网络并保存。

    :param layer_embeddings: {layer_id: np.ndarray[num_tasks, input_dim]}
    :param input_dim: 任务嵌入维度
    :param som_size: SOM 网格大小
    :param num_iterations: SOM 训练轮数
    :param output_dir: 模型保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    for layer_id, embeds in layer_embeddings.items():
        print(f"🚀 Training SOM for Layer {layer_id} ...")
        som = som_network.SOMNetwork(input_dim=input_dim, som_size=som_size, random_seed=layer_id)

        som.train(embeds, num_iterations=num_iterations)

        weight_path = os.path.join(output_dir, f"som_layer_{layer_id}.npy")
        som.save_weights(weight_path)

    print("🎉 All SOM layers trained and saved.")


if __name__ == "__main__":
    num_layers = 4
    input_dim = 384

    # Benchmark-level textual description corpus for SOM pretraining.
    # This corpus is used only to build a semantic topology for subnetwork
    # allocation. It does not include trajectories, rewards, policies, gradients,
    # replay buffers, demonstrations, or environment interaction data.
    description_corpus = SOM_PRETRAIN_DESCRIPTION_CORPUS
    num_descriptions = len(description_corpus)

    task_encoder = SentenceTransformer('all-MiniLM-L12-v2')

    description_embeddings = []
    for description in description_corpus:
        task_e = task_encoder.encode(description)[np.newaxis, :]
        description_embeddings.append(task_e[0])
    description_embeddings = np.array(description_embeddings)

    train_embed = {}
    for layer_id in range(num_layers):
        train_embed[layer_id] = description_embeddings.copy()

    start_time = time.time()
    train_all_layer_soms(
        layer_embeddings=train_embed,
        input_dim=input_dim,
        som_size=(8, 8),
        num_iterations=100,
        output_dir="trained_soms"
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"SOM pretraining corpus size: {num_descriptions}")
    print(f"运行耗时: {elapsed_time:.6f} 秒")
