"""
Hierarchical Clustering for FedH-PSO
层次化聚类：捕捉机构间与患者间的差异
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalClustering:
    """
    层次化聚类器 - 实现两级聚类分析
    """
    
    def __init__(self):
        """初始化层次化聚类器"""
        logger.info("层次化聚类器初始化完成")

    def cluster_institutions(self, local_centers_list: List[np.ndarray],
                             n_clusters: int = 2, linkage: str = 'ward') -> np.ndarray:
        """
        Level 1: 机构层聚类
        对机构上传的聚类中心进行聚类，识别医院分布差异

        Parameters:
        -----------
        local_centers_list : List[np.ndarray]
            各机构的局部聚类中心列表
        n_clusters : int, default=2
            机构聚类的数量
        linkage : str, default='ward'
            链接准则

        Returns:
        --------
        np.ndarray
            机构聚类标签
        """
        logger.info("\n" + "=" * 60)
        logger.info("Level 1: 机构层聚类")
        logger.info("=" * 60)

        n_sites = len(local_centers_list)

        # 使用平均池化，将每个机构的聚类中心平均化，得到固定维度向量
        institution_vectors = []
        for centers in local_centers_list:
            vec = centers.mean(axis=0)  # shape = (n_features,)
            institution_vectors.append(vec)

        institution_vectors = np.vstack(institution_vectors)  # shape = (n_sites, n_features)
        logger.info(f"机构向量矩阵形状: {institution_vectors.shape}")

        # 执行层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        institution_labels = clustering.fit_predict(institution_vectors)

        logger.info(f"机构聚类结果:")
        for i, label in enumerate(institution_labels):
            logger.info(f"  机构 {i + 1} → 聚类 {label}")

        logger.info("=" * 60)

        return institution_labels
    
    def analyze_institution_similarity(self, local_centers_list: List[np.ndarray]) -> np.ndarray:
        """
        分析机构间相似度
        
        Parameters:
        -----------
        local_centers_list : List[np.ndarray]
            各机构的局部聚类中心列表
            
        Returns:
        --------
        np.ndarray
            相似度矩阵 (n_sites, n_sites)
        """
        n_sites = len(local_centers_list)
        similarity_matrix = np.zeros((n_sites, n_sites))
        
        # 计算机构间的距离
        for i in range(n_sites):
            for j in range(i+1, n_sites):
                centers_i = local_centers_list[i]
                centers_j = local_centers_list[j]
                
                # 使用平均Hausdorff距离
                distances = []
                for center_i in centers_i:
                    for center_j in centers_j:
                        dist = np.linalg.norm(center_i - center_j)
                        distances.append(dist)
                
                # 平均距离作为相似度指标（距离越小越相似）
                avg_distance = np.mean(distances)
                similarity = 1.0 / (1.0 + avg_distance)  # 转换为相似度
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # 对角线为1（自己与自己完全相似）
        np.fill_diagonal(similarity_matrix, 1.0)
        
        logger.info("\n机构间相似度矩阵:")
        logger.info(similarity_matrix)
        
        return similarity_matrix
    
    def cluster_patients_with_global_init(self, patient_data: np.ndarray,
                                         global_centers: np.ndarray,
                                         n_clusters: int = None,
                                         max_iter: int = 300) -> Dict:
        """
        Level 2: 患者层聚类
        使用全局聚类中心初始化，执行局部聚类优化
        
        Parameters:
        -----------
        patient_data : np.ndarray
            患者数据 (n_samples, n_features)
        global_centers : np.ndarray
            全局聚类中心
        n_clusters : int, optional
            聚类数量（默认使用global_centers的形状）
        max_iter : int, default=300
            最大迭代次数
            
        Returns:
        --------
        Dict
            包含聚类结果（标签、中心、误差等）
        """
        logger.info("\n" + "="*60)
        logger.info("Level 2: 患者层聚类（使用全局中心初始化）")
        logger.info("="*60)
        
        if n_clusters is None:
            n_clusters = global_centers.shape[0]
        
        n_samples = patient_data.shape[0]
        logger.info(f"患者数量: {n_samples}, 聚类数: {n_clusters}")
        
        # 使用全局中心初始化
        centers = global_centers.copy()
        
        # 执行K-Means迭代（warm-start）
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, init=centers, 
                       n_init=1, max_iter=max_iter, random_state=42)
        labels = kmeans.fit_predict(patient_data)
        
        # 更新后的中心和误差
        updated_centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        
        logger.info(f"K-Means完成: 误差={inertia:.2f}")
        logger.info("="*60)
        
        return {
            'labels': labels,
            'centers': updated_centers,
            'inertia': inertia,
            'n_clusters': n_clusters
        }


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 模拟3个机构的局部聚类中心
    n_clusters = 5
    n_features = 40
    local_centers_list = [
        np.random.rand(n_clusters, n_features),
        np.random.rand(n_clusters, n_features),
        np.random.rand(n_clusters, n_features)
    ]
    
    # 创建层次聚类器
    hc = HierarchicalClustering()
    
    # Level 1: 机构层聚类
    institution_labels = hc.cluster_institutions(local_centers_list, n_clusters=2)
    
    # 分析机构相似度
    similarity_matrix = hc.analyze_institution_similarity(local_centers_list)
    
    print("\n机构聚类标签:", institution_labels)
    print("相似度矩阵:\n", similarity_matrix)

