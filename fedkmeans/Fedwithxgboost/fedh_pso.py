"""
FedH-PSO: Federated Hierarchical Clustering with Particle Swarm Optimization
联邦层次化聚类与粒子群优化框架
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from pso_optimizer import PSOOptimizer
from weighted_aggregation import WeightedAggregator
from hierarchical_clustering import HierarchicalClustering
from scipy.optimize import linear_sum_assignment


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FedHPSO:
    """
    FedH-PSO 主类
    整合PSO优化、加权聚合、层次化聚类三个核心模块
    """
    
    def __init__(self, n_clusters: int = 5, n_features: int = 40,
                 pso_omega: float = 0.7, pso_c1: float = 1.5, pso_c2: float = 1.5,
                 pso_max_iter: int = 100, convergence_threshold: float = 1e-4):
        """
        初始化 FedH-PSO
        
        Parameters:
        -----------
        n_clusters : int, default=5
            聚类数量
        n_features : int, default=40
            特征维度
        pso_omega : float, default=0.7
            PSO惯性权重
        pso_c1 : float, default=1.5
            PSO认知学习因子
        pso_c2 : float, default=1.5
            PSO社会学习因子
        pso_max_iter : int, default=100
            PSO最大迭代次数
        convergence_thre    shold : float, default=1e-4
            全局收敛判据
        """
        self.n_clusters = n_clusters
        self.n_features = n_features
        
        # 创建子模块
        self.pso_optimizer = PSOOptimizer(
            # n_clusters=n_clusters,
            n_clusters_global=n_clusters,
            n_features=n_features,
            omega=pso_omega,
            c1=pso_c1,
            c2=pso_c2,
            max_iter=pso_max_iter
        )
        
        self.weighted_aggregator = WeightedAggregator()
        self.hierarchical_clustering = HierarchicalClustering()
        
        # 收敛判据
        self.convergence_threshold = convergence_threshold
        
        # 存储结果
        self.global_centers = None
        self.weights = None
        self.fitness_history = []
        self.convergence_history = []
        
        logger.info(f"FedH-PSO 初始化完成: K={n_clusters}, 特征={n_features}")

    def federated_clustering(self, upload_infos: List[Dict], 
                           max_global_iter: int = 10) -> Dict:
        """
        执行联邦聚类流程
        
        Parameters:
        -----------
        upload_infos : List[Dict]
            各机构上传的信息，每个元素包含：
            - site_id: 机构ID
            - n_samples: 样本数量
            - inertia: 聚类误差（SSE）
            - cluster_centers: 局部聚类中心
        
        max_global_iter : int, default=10
            最大全局迭代次数
            
        Returns:
        --------
        Dict
            包含全局聚类中心、权重、收敛历史等
        """
        logger.info("\n" + "="*80)
        logger.info("开始 FedH-PSO 联邦聚类流程")
        logger.info("="*80)
        
        n_sites = len(upload_infos)
        
        # 提取局部聚类中心列表
        local_centers_list = [info['cluster_centers'] for info in upload_infos]
        
        # 阶段1：计算机构权重
        logger.info("\n>>> 阶段 1: 计算机构权重")
        weights = self.weighted_aggregator.compute_weights(upload_infos)
        self.weights = weights
        
        # 阶段2：初始加权聚合
        logger.info("\n>>> 阶段 2: 初始加权聚合")
        initial_centers = self.weighted_aggregator.aggregate(upload_infos, weights, k_global=self.n_clusters)
        
        # 迭代优化
        current_centers = initial_centers.copy()
        previous_centers = None
        
        logger.info(f"\n>>> 阶段 3: PSO 优化（最多 {max_global_iter} 次迭代）")
        
        for iter_num in range(max_global_iter):
            logger.info(f"\n--- 全局迭代 {iter_num + 1}/{max_global_iter} ---")
            
            # PSO优化
            optimal_centers, fitness_history = self.pso_optimizer.optimize(
                local_centers_list, weights
            )
            
            # 存储适应度历史
            self.fitness_history.extend(fitness_history)
            
            # 更新全局聚类中心
            previous_centers = current_centers.copy()
            current_centers = optimal_centers.copy()

            # 检查收敛
            if previous_centers is not None:
                def compute_center_diff(current_centers: np.ndarray, previous_centers: np.ndarray) -> float:
                    """
                    计算全局簇中心变化，允许簇数量不同
                    使用Hungarian算法做最优匹配
                    """
                    n_current = current_centers.shape[0]
                    n_prev = previous_centers.shape[0]

                    # 计算距离矩阵，距离越小越匹配
                    distance_matrix = np.linalg.norm(
                        current_centers[:, np.newaxis, :] - previous_centers[np.newaxis, :, :],
                        axis=2
                    )  # shape = (n_current, n_prev)

                    # 最优匹配
                    row_ind, col_ind = linear_sum_assignment(distance_matrix)

                    # 使用匹配计算Frobenius范数
                    matched_diff = current_centers[row_ind] - previous_centers[col_ind]
                    center_diff = np.linalg.norm(matched_diff, ord='fro')

                    return center_diff
                center_diff = compute_center_diff(current_centers, previous_centers)

                self.convergence_history.append(center_diff)
                
                logger.info(f"中心变化: ||C_{{t+1}} - C_t||_F = {center_diff:.6f}")
                
                if center_diff < self.convergence_threshold:
                    logger.info(f"\n✓ 全局聚类中心已收敛（迭代 {iter_num + 1}）")
                    break
            
            # 更新局部中心列表：使用优化后的全局中心作为下一次迭代的参考
            # 这样PSO可以在已优化的基础上继续优化
            local_centers_list = [current_centers.copy() for _ in range(n_sites)]
        
        # 阶段4：层次化聚类分析
        logger.info("\n>>> 阶段 4: 层次化聚类分析")
        
        # Level 1: 机构层聚类
        institution_labels = self.hierarchical_clustering.cluster_institutions(
            [info['cluster_centers'] for info in upload_infos],
            n_clusters=min(3, n_sites)  # 最多分3类
        )
        
        # 机构相似度分析
        similarity_matrix = self.hierarchical_clustering.analyze_institution_similarity(
            [info['cluster_centers'] for info in upload_infos]
        )
        
        # 保存全局聚类中心
        self.global_centers = current_centers
        
        # 构建结果字典
        results = {
            'global_centers': self.global_centers,
            'weights': weights,
            'institution_labels': institution_labels,
            'similarity_matrix': similarity_matrix,
            'fitness_history': self.fitness_history,
            'convergence_history': self.convergence_history,
            'n_iterations': iter_num + 1,
            'converged': iter_num + 1 < max_global_iter
        }
        
        logger.info("\n" + "="*80)
        logger.info("FedH-PSO 流程完成")
        logger.info("="*80)
        
        return results
    
    def local_refinement(self, patient_data: np.ndarray, 
                        global_centers: np.ndarray = None) -> Dict:
        """
        本地细化聚类（Level 2）
        使用全局中心初始化，执行局部聚类优化
        
        Parameters:
        -----------
        patient_data : np.ndarray
            本地患者数据
        global_centers : np.ndarray, optional
            全局聚类中心（默认使用self.global_centers）
            
        Returns:
        --------
        Dict
            本地聚类结果
        """
        if global_centers is None:
            global_centers = self.global_centers
        
        if global_centers is None:
            raise ValueError("全局聚类中心未初始化！请先执行 federated_clustering")
        
        logger.info("\n>>> 本地细化聚类（Level 2）")
        
        result = self.hierarchical_clustering.cluster_patients_with_global_init(
            patient_data, global_centers
        )
        
        return result

    def broadcast_centers(self) -> np.ndarray:
        """
        广播全局聚类中心（供各机构下载）
        
        Returns:
        --------
        np.ndarray
            全局聚类中心矩阵
        """
        if self.global_centers is None:
            raise ValueError("全局聚类中心未初始化！")
        
        logger.info("广播全局聚类中心给各机构")
        return self.global_centers
    
    def get_summary(self) -> Dict:
        """
        获取流程总结信息
        
        Returns:
        --------
        Dict
            总结信息
        """
        summary = {
            'n_clusters': self.n_clusters,
            'n_features': self.n_features,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'convergence_history': self.convergence_history,
            'final_fitness': self.fitness_history[-1] if self.fitness_history else None,
            'n_iterations': len(self.convergence_history)
        }
        
        return summary


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 模拟3个机构的上传信息
    upload_infos = []
    for site_id in [1, 2, 3]:
        n_samples = np.random.randint(8000, 10000)
        inertia = np.random.uniform(100, 200)
        centers = np.random.rand(5, 40)
        
        upload_infos.append({
            'site_id': site_id,
            'n_samples': n_samples,
            'inertia': inertia,
            'cluster_centers': centers
        })
    
    # 创建FedH-PSO
    fedh_pso = FedHPSO(n_clusters=5, n_features=40, pso_max_iter=20)
    
    # 执行联邦聚类
    results = fedh_pso.federated_clustering(upload_infos, max_global_iter=3)
    
    # 打印结果
    print(f"\n全局聚类中心形状: {results['global_centers'].shape}")
    print(f"机构权重: {results['weights']}")
    print(f"机构聚类标签: {results['institution_labels']}")
    print(f"是否收敛: {results['converged']}")

