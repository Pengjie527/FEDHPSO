import numpy as np
import logging
from typing import List, Dict
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedAggregator:
    """
    加权聚合器 - 支持局部簇数量可变
    """

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def compute_weights(self, upload_infos: List[Dict], alpha: float = 0.5) -> np.ndarray:
        n_sites = len(upload_infos)
        total_samples = sum(info['n_samples'] for info in upload_infos)
        total_inertia = sum(info.get('inertia', 1.0) for info in upload_infos)

        weights = []
        for info in upload_infos:
            n_samples = info['n_samples']
            sse = info.get('inertia', 1.0)
            cluster_var = np.var(info['cluster_centers'], axis=0).mean()
            severity = info.get('avg_severity', 0.0)

            base_weight = (n_samples / total_samples) / (sse / total_inertia + self.epsilon)
            weight = base_weight / (cluster_var + self.epsilon) * (1 + alpha * severity)
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / np.sum(weights)
        return weights

    def aggregate(self, upload_infos: List[Dict], weights: np.ndarray, k_global: int) -> np.ndarray:
        """
        使用权重对聚类中心进行加权聚合，允许局部簇数量不同
        """
        n_features = upload_infos[0]['cluster_centers'].shape[1]
        global_centers = np.zeros((k_global, n_features))

        for i, info in enumerate(upload_infos):
            local_centers = info['cluster_centers']
            n_local = local_centers.shape[0]
            weight = weights[i]

            if n_local != k_global:
                # 使用Hungarian算法做最近邻匹配
                distance_matrix = np.linalg.norm(
                    local_centers[:, np.newaxis, :] - global_centers[np.newaxis, :, :],
                    axis=2
                )
                row_ind, col_ind = linear_sum_assignment(distance_matrix)
                matched_centers = np.zeros((k_global, n_features))
                for r, c in zip(row_ind, col_ind):
                    matched_centers[c] = local_centers[r]
            else:
                matched_centers = local_centers

            global_centers += weight * matched_centers

        logger.info(f"加权聚合完成: 全局聚类中心形状 = {global_centers.shape}")
        return global_centers


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    upload_infos = []
    for site_id in [1, 2, 3]:
        n_samples = np.random.randint(8000, 10000)
        inertia = np.random.uniform(100, 200)
        centers = np.random.rand(np.random.randint(4, 6), 40)  # 局部簇数可变
        upload_infos.append({
            'site_id': site_id,
            'n_samples': n_samples,
            'inertia': inertia,
            'cluster_centers': centers
        })

    aggregator = WeightedAggregator()
    weights = aggregator.compute_weights(upload_infos)
    global_centers = aggregator.aggregate(upload_infos, weights, k_global=5)

    print(f"聚合后的全局聚类中心形状: {global_centers.shape}")
    print(f"权重和为: {np.sum(weights):.6f}")
