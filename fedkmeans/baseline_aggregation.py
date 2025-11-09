"""
Step 1: Baseline Weighted Average Aggregation
基线方法：加权平均聚合
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CentralServer:
    """中心服务器类 - 模拟联邦学习中的中心协调者"""
    
    def __init__(self, n_clusters: int = 750):
        """
        初始化中心服务器
        
        Parameters:
        -----------
        n_clusters : int
            聚类数量
        """
        self.n_clusters = n_clusters
        self.global_centers = None
        self.site_infos = []
        
        logger.info(f"中心服务器初始化完成 (K={n_clusters})")
    
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
    
    def aggregate_weighted_average(self) -> np.ndarray:
        """
        基线方法：加权平均聚合
        
        使用每个站点的样本数量作为权重，对聚类中心进行加权平均
        
        Returns:
        --------
        np.ndarray : 全局聚类中心 (n_clusters, n_features)
        """
        logger.info("\n" + "="*60)
        logger.info("开始基线方法：加权平均聚合")
        logger.info("="*60)
        
        # 获取特征维度
        n_features = self.site_infos[0]['cluster_centers'].shape[1]
        
        # 初始化全局聚类中心
        self.global_centers = np.zeros((self.n_clusters, n_features))
        
        # 计算总样本数
        total_samples = sum(info['n_samples'] for info in self.site_infos)
        
        logger.info(f"总样本数: {total_samples}")
        logger.info("各站点权重:")
        
        # 对每个簇进行加权平均
        for info in self.site_infos:
            site_id = info['site_id']
            centers = info['cluster_centers']
            n_samples = info['n_samples']
            weight = n_samples / total_samples
            
            logger.info(f"  站点 {site_id}: 样本数={n_samples}, 权重={weight:.4f}")
            
            # 加权累加
            self.global_centers += weight * centers
        
        logger.info(f"\n全局聚类中心生成完成: {self.global_centers.shape}")
        logger.info("="*60)
        
        return self.global_centers
    
    def aggregate_cluster_size_weighted(self) -> np.ndarray:
        """
        改进的加权平均方法：考虑簇大小的加权
        
        对于每个聚类中心，使用该簇在各站点的样本数量作为权重
        
        Returns:
        --------
        np.ndarray : 全局聚类中心 (n_clusters, n_features)
        """
        logger.info("\n" + "="*60)
        logger.info("开始改进方法：簇大小加权聚合")
        logger.info("="*60)
        
        # 获取特征维度
        n_features = self.site_infos[0]['cluster_centers'].shape[1]
        
        # 初始化全局聚类中心
        self.global_centers = np.zeros((self.n_clusters, n_features))
        
        # 对每个簇分别计算加权平均
        for k in range(self.n_clusters):
            cluster_weight_sum = 0
            cluster_center_weighted = np.zeros(n_features)
            
            for info in self.site_infos:
                centers = info['cluster_centers']
                cluster_sizes = info['cluster_sizes']
                
                # 第k个簇的大小作为权重
                weight = cluster_sizes[k]
                cluster_weight_sum += weight
                cluster_center_weighted += weight * centers[k]
            
            # 归一化
            if cluster_weight_sum > 0:
                self.global_centers[k] = cluster_center_weighted / cluster_weight_sum
            else:
                # 如果所有站点该簇都为空，使用简单平均
                self.global_centers[k] = np.mean(
                    [info['cluster_centers'][k] for info in self.site_infos],
                    axis=0
                )
        
        # 统计非空簇
        non_empty_clusters = 0
        for k in range(self.n_clusters):
            total_size = sum(info['cluster_sizes'][k] for info in self.site_infos)
            if total_size > 0:
                non_empty_clusters += 1
        
        logger.info(f"非空簇数量: {non_empty_clusters}/{self.n_clusters}")
        logger.info(f"全局聚类中心生成完成: {self.global_centers.shape}")
        logger.info("="*60)
        
        return self.global_centers
    
    def save_global_centers(self, output_path: str):
        """
        保存全局聚类中心
        
        Parameters:
        -----------
        output_path : str
            输出文件路径
        """
        if self.global_centers is None:
            raise ValueError("尚未生成全局聚类中心")
        
        np.save(output_path, self.global_centers)
        logger.info(f"全局聚类中心已保存至: {output_path}")
    
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


def evaluate_global_clustering(sites: List, global_centers: np.ndarray) -> pd.DataFrame:
    """
    评估全局聚类在各站点的性能
    
    Parameters:
    -----------
    sites : List[LocalSite]
        本地站点列表
    global_centers : np.ndarray
        全局聚类中心
        
    Returns:
    --------
    pd.DataFrame : 评估结果
    """
    logger.info("\n" + "="*60)
    logger.info("评估全局聚类性能")
    logger.info("="*60)
    
    results = []
    
    for site in sites:
        logger.info(f"\n评估站点 {site.site_id}...")
        
        # 计算本地惯性（使用本地聚类中心）
        local_inertia = site.local_model.inertia_
        
        # 计算全局惯性（使用全局聚类中心）
        from scipy.spatial.distance import cdist
        distances = cdist(site.features, global_centers, metric='euclidean')
        global_inertia = np.sum(np.min(distances, axis=1) ** 2)
        
        # 计算聚合质量比率（原来的"惯性比率"）
        # 比率越小越好，表示全局聚合的效果接近本地最优
        aggregation_quality_ratio = global_inertia / local_inertia
        
        result = {
            'site_id': site.site_id,
            'n_samples': len(site.data),
            'local_inertia': local_inertia,
            'global_inertia': global_inertia,
            'inertia_ratio': aggregation_quality_ratio,  # 保持向后兼容
            'aggregation_quality': aggregation_quality_ratio,  # 新名称
            'performance_degradation': (aggregation_quality_ratio - 1) * 100  # 性能下降百分比
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
    
    logger.info("\n总体统计:")
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
    
    # 创建中心服务器
    server = CentralServer(n_clusters=750)
    server.receive_uploads(upload_infos)
    
    # 方法1：简单加权平均
    global_centers_1 = server.aggregate_weighted_average()
    results_1 = evaluate_global_clustering(sites, global_centers_1)
    print("\n方法1 (简单加权平均) 评估结果:")
    print(results_1)
    
    # 方法2：簇大小加权
    global_centers_2 = server.aggregate_cluster_size_weighted()
    results_2 = evaluate_global_clustering(sites, global_centers_2)
    print("\n方法2 (簇大小加权) 评估结果:")
    print(results_2)

