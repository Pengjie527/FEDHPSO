"""
Step 2: Advanced Aggregation Methods
改进的聚合方法：匈牙利匹配、层次聚类、联邦迭代优化
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedCentralServer:
    """改进的中心服务器类 - 支持多种高级聚合方法"""
    
    def __init__(self, n_clusters: int = 750):
        """
        初始化改进的中心服务器
        
        Parameters:
        -----------
        n_clusters : int
            聚类数量
        """
        self.n_clusters = n_clusters
        self.global_centers = None
        self.site_infos = []
        
        logger.info(f"改进中心服务器初始化完成 (K={n_clusters})")
    
    def receive_uploads(self, upload_infos: List[Dict]):
        """
        接收各站点上传的信息
        
        Parameters:
        -----------
        upload_infos : List[Dict]
            各站点上传的信息列表
        """
        self.site_infos = upload_infos
        
        logger.info(f"中心服务器接收到 {len(upload_infos)} 个站点的上传信息")
        for info in upload_infos:
            logger.info(f"  站点 {info['site_id']}: {info['n_samples']} 样本, "
                       f"{info['cluster_centers'].shape[0]} 个聚类中心")
    
    def aggregate_with_hungarian_matching(self) -> np.ndarray:
        """
        方法1: 匈牙利匹配聚合
        
        使用匈牙利算法找到各站点聚类中心的最优匹配，然后进行加权聚合
        
        Returns:
        --------
        np.ndarray : 全局聚类中心 (n_clusters, n_features)
        """
        logger.info("\n" + "="*60)
        logger.info("开始方法1: 匈牙利匹配聚合")
        logger.info("="*60)
        
        if len(self.site_infos) < 2:
            raise ValueError("至少需要2个站点才能进行匹配")
        
        # Step 1: 以第一个站点为基准，匹配其他站点
        reference_centers = self.site_infos[0]['cluster_centers']
        aligned_centers = [reference_centers]
        aligned_sizes = [self.site_infos[0]['cluster_sizes']]
        
        logger.info(f"以站点 {self.site_infos[0]['site_id']} 为基准进行匹配")
        
        for i in range(1, len(self.site_infos)):
            site_centers = self.site_infos[i]['cluster_centers']
            site_sizes = self.site_infos[i]['cluster_sizes']
            
            logger.info(f"匹配站点 {self.site_infos[i]['site_id']}...")
            
            # 计算距离矩阵 (n_clusters × n_clusters)
            distances = cdist(reference_centers, site_centers, 'euclidean')
            
            # 匈牙利算法求解最优匹配
            row_indices, col_indices = linear_sum_assignment(distances)
            
            # 重新排列聚类中心和簇大小
            aligned_center = site_centers[col_indices]
            aligned_size = site_sizes[col_indices]
            
            aligned_centers.append(aligned_center)
            aligned_sizes.append(aligned_size)
            
            # 计算匹配质量
            total_distance = np.sum(distances[row_indices, col_indices])
            avg_distance = total_distance / len(row_indices)
            logger.info(f"  匹配完成，平均距离: {avg_distance:.4f}")
        
        # Step 2: 加权聚合（考虑簇大小）
        n_features = reference_centers.shape[1]
        self.global_centers = np.zeros((self.n_clusters, n_features))
        
        logger.info("开始加权聚合...")
        
        for k in range(self.n_clusters):
            weighted_sum = np.zeros(n_features)
            total_weight = 0
            
            for i, (center, sizes) in enumerate(zip(aligned_centers, aligned_sizes)):
                cluster_size = sizes[k]
                weight = cluster_size
                
                weighted_sum += weight * center[k]
                total_weight += weight
            
            if total_weight > 0:
                self.global_centers[k] = weighted_sum / total_weight
            else:
                # 如果所有站点该簇都为空，使用简单平均
                self.global_centers[k] = np.mean([aligned_centers[i][k] 
                                                for i in range(len(aligned_centers))], axis=0)
        
        # 统计非空簇
        non_empty_clusters = 0
        for k in range(self.n_clusters):
            total_size = sum(aligned_sizes[i][k] for i in range(len(aligned_sizes)))
            if total_size > 0:
                non_empty_clusters += 1
        
        logger.info(f"匈牙利匹配聚合完成:")
        logger.info(f"  - 非空簇数量: {non_empty_clusters}/{self.n_clusters}")
        logger.info(f"  - 全局聚类中心形状: {self.global_centers.shape}")
        logger.info("="*60)
        
        return self.global_centers
    
    def aggregate_hierarchical_clustering(self) -> np.ndarray:
        """
        方法2: 层次聚类聚合
        
        将所有站点的聚类中心合并，使用层次聚类重新分组，然后计算质心
        
        Returns:
        --------
        np.ndarray : 全局聚类中心 (n_clusters, n_features)
        """
        logger.info("\n" + "="*60)
        logger.info("开始方法2: 层次聚类聚合")
        logger.info("="*60)
        
        # 合并所有站点的聚类中心
        all_centers = np.vstack([info['cluster_centers'] for info in self.site_infos])
        all_sizes = np.concatenate([info['cluster_sizes'] for info in self.site_infos])
        
        logger.info(f"合并后中心数量: {len(all_centers)} (各站点: {[info['site_id'] for info in self.site_infos]})")
        
        # 层次聚类
        logger.info("执行层次聚类...")
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='ward'
        )
        cluster_labels = clustering.fit_predict(all_centers)
        
        # 计算每个簇的加权质心
        logger.info("计算加权质心...")
        global_centers = []
        
        for k in range(self.n_clusters):
            mask = (cluster_labels == k)
            if np.sum(mask) > 0:
                centers_in_cluster = all_centers[mask]
                sizes_in_cluster = all_sizes[mask]
                
                # 加权平均
                if np.sum(sizes_in_cluster) > 0:
                    weights = sizes_in_cluster / np.sum(sizes_in_cluster)
                    weighted_center = np.average(centers_in_cluster, axis=0, weights=weights)
                else:
                    weighted_center = np.mean(centers_in_cluster, axis=0)
                global_centers.append(weighted_center)
            else:
                # 空簇处理
                global_centers.append(np.mean(all_centers, axis=0))
        
        self.global_centers = np.array(global_centers)
        
        # 统计每个全局簇包含的原始中心数量
        cluster_counts = [np.sum(cluster_labels == k) for k in range(self.n_clusters)]
        logger.info(f"层次聚类聚合完成:")
        logger.info(f"  - 全局聚类中心形状: {self.global_centers.shape}")
        logger.info(f"  - 平均每个全局簇包含原始中心: {np.mean(cluster_counts):.2f}")
        logger.info(f"  - 最大簇包含中心数: {max(cluster_counts)}")
        logger.info("="*60)
        
        return self.global_centers
    
    def aggregate_federated_iterative(self, sites: List, max_iter: int = 20, 
                                    convergence_threshold: float = 1e-4,
                                    log_every: int = 5) -> np.ndarray:
        """
        方法3: 联邦迭代优化
        
        使用联邦版本的Lloyd算法，迭代优化全局聚类中心
        
        Parameters:
        -----------
        sites : List[LocalSite]
            本地站点列表
        max_iter : int
            最大迭代次数
        convergence_threshold : float
            收敛阈值
        log_every : int
            每隔多少次迭代记录一次详细信息（避免日志过多）
            
        Returns:
        --------
        np.ndarray : 全局聚类中心 (n_clusters, n_features)
        """
        logger.info("\n" + "="*60)
        logger.info("开始方法3: 联邦迭代优化")
        logger.info("="*60)
        
        # 初始化全局中心（使用第一个站点的中心作为初始值）
        self.global_centers = self.site_infos[0]['cluster_centers'].copy()
        
        logger.info(f"初始全局中心形状: {self.global_centers.shape}")
        logger.info(f"最大迭代次数: {max_iter}, 收敛阈值: {convergence_threshold}")
        
        for iteration in range(max_iter):
            # 只记录关键迭代或每 log_every 次迭代
            should_log = (iteration == 0) or (iteration % log_every == log_every - 1) or (iteration == max_iter - 1)
            if should_log:
                logger.info(f"迭代 {iteration + 1}/{max_iter}")
            
            # 各站点用当前全局中心重新分配样本
            new_site_centers = []
            
            for site in sites:
                # 计算每个样本到所有全局中心的距离
                distances = cdist(site.features, self.global_centers, 'euclidean')
                assignments = np.argmin(distances, axis=1)
                
                # 计算新的局部中心
                site_centers = []
                for k in range(self.n_clusters):
                    mask = (assignments == k)
                    if np.sum(mask) > 0:
                        center = np.mean(site.features[mask], axis=0)
                    else:
                        center = self.global_centers[k]  # 保持原中心
                    site_centers.append(center)
                new_site_centers.append(np.array(site_centers))
            
            # 聚合所有站点的中心
            prev_centers = self.global_centers.copy()
            self.global_centers = np.mean(new_site_centers, axis=0)
            
            # 计算中心变化量
            center_change = np.mean(np.abs(self.global_centers - prev_centers))
            
            # 只在需要记录时才输出详情
            if should_log:
                logger.info(f"  中心变化量: {center_change:.6f}")
            
            # 检查收敛
            if center_change < convergence_threshold:
                logger.info(f"在第 {iteration + 1} 次迭代收敛")
                break
        
        logger.info(f"联邦迭代优化完成:")
        logger.info(f"  - 实际迭代次数: {iteration + 1}")
        logger.info(f"  - 最终中心变化量: {center_change:.6f}")
        logger.info(f"  - 全局聚类中心形状: {self.global_centers.shape}")
        logger.info("="*60)
        
        return self.global_centers
    
    def get_aggregation_statistics(self) -> Dict:
        """
        获取聚合统计信息
        
        Returns:
        --------
        Dict : 统计信息
        """
        if self.global_centers is None:
            raise ValueError("尚未生成全局聚类中心")
        
        stats = {
            'n_sites': len(self.site_infos),
            'n_clusters': self.n_clusters,
            'n_features': self.global_centers.shape[1],
            'total_samples': sum(info['n_samples'] for info in self.site_infos),
            'site_samples': {info['site_id']: info['n_samples'] 
                           for info in self.site_infos}
        }
        
        # 计算全局聚类中心的统计量
        stats['global_center_mean'] = np.mean(self.global_centers)
        stats['global_center_std'] = np.std(self.global_centers)
        stats['global_center_min'] = np.min(self.global_centers)
        stats['global_center_max'] = np.max(self.global_centers)
        
        return stats


def evaluate_advanced_clustering(sites: List, global_centers: np.ndarray, 
                               method_name: str) -> pd.DataFrame:
    """
    评估改进聚合方法的性能
    
    Parameters:
    -----------
    sites : List[LocalSite]
        本地站点列表
    global_centers : np.ndarray
        全局聚类中心
    method_name : str
        方法名称
        
    Returns:
    --------
    pd.DataFrame : 评估结果
    """
    logger.info(f"\n评估 {method_name} 性能")
    logger.info("="*60)
    
    results = []
    
    for site in sites:
        logger.info(f"评估站点 {site.site_id}...")
        
        # 计算本地惯性（使用本地聚类中心）
        local_inertia = site.local_model.inertia_
        
        # 计算全局惯性（使用全局聚类中心）
        distances = cdist(site.features, global_centers, metric='euclidean')
        global_inertia = np.sum(np.min(distances, axis=1) ** 2)
        
        # 计算聚合质量比率（原来的"惯性比率"）
        aggregation_quality_ratio = global_inertia / local_inertia
        
        result = {
            'site_id': site.site_id,
            'n_samples': len(site.data),
            'local_inertia': local_inertia,
            'global_inertia': global_inertia,
            'inertia_ratio': aggregation_quality_ratio,  # 保持向后兼容
            'aggregation_quality': aggregation_quality_ratio,  # 新名称
            'performance_degradation': (aggregation_quality_ratio - 1) * 100
        }
        
        results.append(result)
        
        logger.info(f"  本地聚类误差: {local_inertia:.2f}")
        logger.info(f"  全局聚类误差: {global_inertia:.2f}")
        logger.info(f"  聚合质量比: {aggregation_quality_ratio:.4f} (越小越好)")
        logger.info(f"  性能下降: {result['performance_degradation']:.2f}%")
    
    # 计算总体统计
    total_local_inertia = sum(r['local_inertia'] for r in results)
    total_global_inertia = sum(r['global_inertia'] for r in results)
    overall_ratio = total_global_inertia / total_local_inertia
    
    logger.info(f"\n{method_name} 总体统计:")
    logger.info(f"  总本地聚类误差: {total_local_inertia:.2f}")
    logger.info(f"  总全局聚类误差: {total_global_inertia:.2f}")
    logger.info(f"  总体聚合质量比: {overall_ratio:.4f} (越小越好)")
    logger.info(f"  总体性能下降: {(overall_ratio - 1) * 100:.2f}%")
    logger.info("="*60)
    
    df_results = pd.DataFrame(results)
    
    # 添加总体行
    total_row = {
        'site_id': 'Total',
        'n_samples': sum(r['n_samples'] for r in results),
        'local_inertia': total_local_inertia,
        'global_inertia': total_global_inertia,
        'inertia_ratio': overall_ratio,  # 保持向后兼容
        'aggregation_quality': overall_ratio,  # 新名称
        'performance_degradation': (overall_ratio - 1) * 100
    }
    df_results = pd.concat([df_results, pd.DataFrame([total_row])], ignore_index=True)
    
    return df_results


if __name__ == "__main__":
    # 测试代码
    from local_clustering import run_local_clustering_all_sites
    
    data_paths = {
        1: "Dataset/dataave/group1_mimic_data.csv",
        2: "Dataset/dataave/group2_mimic_data.csv",
        3: "Dataset/dataave/group3_mimic_data.csv"
    }
    
    # 执行本地聚类
    sites, upload_infos = run_local_clustering_all_sites(
        data_paths=data_paths,
        n_clusters=750,
        random_state=42
    )
    
    # 创建改进的中心服务器
    server = AdvancedCentralServer(n_clusters=750)
    server.receive_uploads(upload_infos)
    
    # 方法1：匈牙利匹配
    global_centers_1 = server.aggregate_with_hungarian_matching()
    results_1 = evaluate_advanced_clustering(sites, global_centers_1, "匈牙利匹配")
    print("\n方法1 (匈牙利匹配) 评估结果:")
    print(results_1)
    
    # 方法2：层次聚类
    global_centers_2 = server.aggregate_hierarchical_clustering()
    results_2 = evaluate_advanced_clustering(sites, global_centers_2, "层次聚类")
    print("\n方法2 (层次聚类) 评估结果:")
    print(results_2)
    
    # 方法3：联邦迭代
    global_centers_3 = server.aggregate_federated_iterative(sites, max_iter=10)
    results_3 = evaluate_advanced_clustering(sites, global_centers_3, "联邦迭代")
    print("\n方法3 (联邦迭代) 评估结果:")
    print(results_3)
