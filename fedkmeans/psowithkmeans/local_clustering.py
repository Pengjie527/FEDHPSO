"""
Step 1: Local K-Means Clustering for Each Site
每个站点本地执行K-Means聚类
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, Tuple, List
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import sys
import os
# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from featuredatameasure.preprocessing import (
    preprocess_data,
    apply_log_transform,
    standardize_features,
    get_feature_definitions,
    PreprocessingConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalSite:
    """本地站点类 - 模拟每个医疗机构的本地数据和计算"""
    
    def __init__(self, site_id: int, data_path: str, n_clusters: int = 750):
        """
        初始化本地站点
        
        Parameters:
        -----------
        site_id : int
            站点编号 (1, 2, 3)
        data_path : str
            本地数据文件路径
        n_clusters : int
            聚类数量，默认750
        """
        self.site_id = site_id
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.data = None
        self.features = None
        self.scaler = StandardScaler()
        self.local_model = None
        self.cluster_centers = None
        self.cluster_sizes = None
        self.labels = None
        
        logger.info(f"站点 {site_id} 初始化完成")
    
    def load_data(self, feature_columns: List[str] = None, use_first_bloc_only: bool = True):
        """
        加载本地数据
        
        Parameters:
        -----------
        feature_columns : List[str], optional
            需要使用的特征列，如果为None则使用所有数值列
        use_first_bloc_only : bool, default=True
            是否只使用每个患者的第一条bloc数据
        """
        # 创建预处理配置
        config = PreprocessingConfig(
            use_first_bloc_only=use_first_bloc_only,
            feature_columns=feature_columns
        )
        
        # 使用预处理模块加载和预处理数据
        self.data, self.features, feature_columns_used, self.scaler = preprocess_data(
            data_path=self.data_path,
            config=config,
            site_id=self.site_id,
            logger_instance=logger
        )

        return self.features
    
    def normalize_data(self):
        """标准化数据
        
        - 二进制特征: 已经 -0.5 处理过
        - 正态分布特征: Z-score标准化
        - 对数正态特征: log(0.1+x) 然后 Z-score标准化
        """
        logger.info(f"站点 {self.site_id} 正在标准化数据...")

        # 应用对数变换（对对数正态特征）
        self.features = apply_log_transform(self.features, logger_instance=logger)
        
        # 标准化特征（使用已存在的scaler，但重新拟合）
        self.features, self.scaler = standardize_features(
            self.features,
            scaler=self.scaler,
            fit_scaler=True,
            logger_instance=logger
        )

        return self.features

    
    def determine_optimal_clusters(self, method: str = 'improved_heuristic', 
                                    min_k: int = 2, max_k: int = 50,
                                    test_k_range: tuple = None):
        """
        确定最优聚类数量
        
        Parameters:
        -----------
        method : str, default='improved_heuristic'
            选择方法：
            - 'improved_heuristic': 改进的启发式规则（快速，推荐）
            - 'elbow': 肘部法则（较慢但更准确）
            - 'silhouette': 轮廓系数法（最慢但最准确）
        min_k : int, default=2
            最小聚类数量
        max_k : int, default=50
            最大聚类数量（避免过拟合）
        test_k_range : tuple, optional
            测试的k值范围，格式为 (min_k, max_k, step)
            如果为None，则根据数据自动确定
            
        Returns:
        --------
        int : 最优聚类数量
        """
        n_samples = len(self.data)
        n_features = len(self.features.columns)
        
        if method == 'improved_heuristic':
            # 改进的启发式规则
            # 1. 基于样本数：使用平方根规则，但更保守
            k_samples = max(2, int(np.sqrt(n_samples)))
            
            # 2. 基于特征数：使用对数或平方根，而不是直接乘以2
            # 特征数不应该直接决定簇数，而是使用更温和的函数
            k_features = max(2, int(np.sqrt(n_features)) * 2)  # 使用平方根后乘以2
            
            # 3. 综合规则：取两者中较小的值（更保守）
            k = min(k_samples, k_features)
            
            # 4. 应用上下限
            k = max(min_k, min(k, max_k))
            
            logger.info(f"  启发式规则计算结果: k_samples={k_samples}, k_features={k_features}, 最终k={k}")
            
        elif method == 'elbow':
            # 肘部法则：寻找inertia下降的"肘部"点
            if test_k_range is None:
                # 自动确定测试范围
                heuristic_k = self.determine_optimal_clusters(method='improved_heuristic', 
                                                             min_k=min_k, max_k=max_k)
                test_min = max(min_k, heuristic_k - 5)
                test_max = min(max_k, heuristic_k + 10)
                k_range = range(test_min, test_max + 1, max(1, (test_max - test_min) // 10))
            else:
                k_range = range(*test_k_range)
            
            logger.info(f"  肘部法则: 测试k值范围 {min(k_range)}-{max(k_range)}...")
            inertias = []
            k_values = list(k_range)
            
            for k in k_values:
                model = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
                model.fit(self.features)
                inertias.append(model.inertia_)
            
            # 计算inertia的相对下降率
            # 寻找下降率最大的点（肘部）
            if len(inertias) >= 2:
                decreases = []
                for i in range(1, len(inertias)):
                    # 计算相对下降率
                    relative_decrease = (inertias[i-1] - inertias[i]) / inertias[i-1] if inertias[i-1] > 0 else 0
                    decreases.append(relative_decrease)
                
                # 找到下降率显著变小的点（肘部）
                if len(decreases) >= 2:
                    # 计算下降率的变化
                    decrease_changes = np.diff(decreases)
                    # 找到第一个下降变化小于平均值的点
                    elbow_idx = np.where(decrease_changes < np.mean(decrease_changes))[0]
                    if len(elbow_idx) > 0:
                        k = k_values[elbow_idx[0] + 1]
                    else:
                        # 如果没有明显肘部，选择inertia下降率最大的点
                        k = k_values[np.argmax(decreases) + 1]
                else:
                    k = k_values[0]
            else:
                k = k_values[0]
            
            k = max(min_k, min(k, max_k))
            logger.info(f"  肘部法则结果: k={k}")
            
        elif method == 'silhouette':
            # 轮廓系数法：选择轮廓系数最高的k
            if test_k_range is None:
                heuristic_k = self.determine_optimal_clusters(method='improved_heuristic', 
                                                             min_k=min_k, max_k=max_k)
                test_min = max(min_k, heuristic_k - 3)
                test_max = min(max_k, heuristic_k + 7)
                k_range = range(test_min, test_max + 1, 2)  # 步长为2以加快速度
            else:
                k_range = range(*test_k_range)
            
            logger.info(f"  轮廓系数法: 测试k值范围 {min(k_range)}-{max(k_range)}...")
            
            from sklearn.metrics import silhouette_score
            
            # 如果样本太多，只使用部分样本计算轮廓系数
            max_samples_for_silhouette = 5000
            if len(self.data) > max_samples_for_silhouette:
                sample_indices = np.random.choice(len(self.data), max_samples_for_silhouette, replace=False)
                # 处理 DataFrame 和 numpy 数组两种情况
                if isinstance(self.features, pd.DataFrame):
                    features_sample = self.features.iloc[sample_indices]
                else:
                    features_sample = self.features[sample_indices]
            else:
                features_sample = self.features
            
            # 确保是 numpy 数组格式（用于 silhouette_score）
            if isinstance(features_sample, pd.DataFrame):
                features_sample = features_sample.values
            
            best_k = min_k
            best_score = -1
            
            for k in k_range:
                model = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
                labels = model.fit_predict(features_sample)
                
                if len(np.unique(labels)) > 1:  # 确保有多个簇
                    score = silhouette_score(features_sample, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            
            k = max(min_k, min(best_k, max_k))
            logger.info(f"  轮廓系数法结果: k={k} (轮廓系数={best_score:.3f})")
        else:
            raise ValueError(f"未知的方法: {method}")
        
        return k


    def perform_local_clustering(self, random_state: int = 42, auto_determine_k: bool = True,
                                 k_method: str = 'improved_heuristic', min_k: int = 2, 
                                 max_k: int = 50, test_k_range: tuple = None):
        """
        执行本地K-Means聚类
        
        Parameters:
        -----------
        random_state : int
            随机种子
        auto_determine_k : bool, default=True
            是否自动确定聚类数量
        k_method : str, default='improved_heuristic'
            确定聚类数量的方法（见 determine_optimal_clusters 文档）
        min_k : int, default=2
            最小聚类数量
        max_k : int, default=50
            最大聚类数量
        test_k_range : tuple, optional
            测试的k值范围（仅用于 elbow 和 silhouette 方法）
            
        Returns:
        --------
        Dict : 包含聚类结果的字典
        """
        # 自动确定聚类数量
        if auto_determine_k:
            optimal_k = self.determine_optimal_clusters(
                method=k_method,
                min_k=min_k,
                max_k=max_k,
                test_k_range=test_k_range
            )
            self.n_clusters = optimal_k
            logger.info(f"站点 {self.site_id} 使用自动确定的聚类数量: {self.n_clusters} (方法: {k_method})")
        
        logger.info(f"站点 {self.site_id} 开始本地聚类 (K={self.n_clusters})...")
        
        # 执行K-Means聚类
        self.local_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        self.labels = self.local_model.fit_predict(self.features)
        self.cluster_centers = self.local_model.cluster_centers_
        
        # 计算每个簇的大小
        unique, counts = np.unique(self.labels, return_counts=True)
        self.cluster_sizes = np.zeros(self.n_clusters)
        self.cluster_sizes[unique] = counts
        
        # 计算聚类质量指标
        inertia = self.local_model.inertia_
        
        logger.info(f"站点 {self.site_id} 本地聚类完成")
        logger.info(f"  - 聚类数量: {self.n_clusters}")
        logger.info(f"  - 惯性 (Inertia): {inertia:.2f}")
        logger.info(f"  - 非空簇数量: {np.sum(self.cluster_sizes > 0)}/{self.n_clusters}")
        logger.info(f"  - 平均簇大小: {np.mean(self.cluster_sizes[self.cluster_sizes > 0]):.2f}")
        
        return {
            'site_id': self.site_id,
            'cluster_centers': self.cluster_centers,
            'cluster_sizes': self.cluster_sizes,
            'labels': self.labels,
            'inertia': inertia,
            'n_samples': len(self.data),
            'n_clusters': self.n_clusters
        }
    
    def get_upload_info(self) -> Dict:
        """
        获取需要上传到中心服务器的信息
        在实际联邦学习中，这些信息会被加密传输
        
        Returns:
        --------
        Dict : 包含聚类中心和簇大小的字典
        """
        if self.cluster_centers is None:
            raise ValueError(f"站点 {self.site_id} 尚未执行聚类")
        
        upload_info = {
            'site_id': self.site_id,
            'cluster_centers': self.cluster_centers.copy(),
            'cluster_sizes': self.cluster_sizes.copy(),
            'n_samples': len(self.data),
            'n_features': self.cluster_centers.shape[1]
        }
        
        logger.info(f"站点 {self.site_id} 准备上传信息:")
        logger.info(f"  - 聚类中心: {upload_info['cluster_centers'].shape}")
        logger.info(f"  - 簇大小: {len(upload_info['cluster_sizes'])}")
        logger.info(f"  - 样本数: {upload_info['n_samples']}")
        
        return upload_info
    
    def evaluate_clustering(self, global_centers: np.ndarray = None) -> Dict:
        """
        评估聚类质量
        
        Parameters:
        -----------
        global_centers : np.ndarray, optional
            全局聚类中心，用于比较
            
        Returns:
        --------
        Dict : 评估指标，包括:
            - silhouette_score: 轮廓系数 (越大越好，范围: -1 到 1)
            - calinski_harabasz_score: Calinski-Harabasz指数 (越大越好)
            - davies_bouldin_score: Davies-Bouldin指数 (越小越好)
            - inertia: 聚类惯性 (越小越好)
        """
        from sklearn.metrics import (
            silhouette_score, 
            calinski_harabasz_score,
            davies_bouldin_score
        )
        
        metrics = {
            'site_id': self.site_id,
            'inertia': self.local_model.inertia_,
            'n_clusters': self.n_clusters,
            'n_samples': len(self.data),
        }
        
        # 确保特征和标签是numpy数组格式
        features_array = self.features.values if hasattr(self.features, 'values') else self.features
        labels_array = self.labels if isinstance(self.labels, np.ndarray) else np.array(self.labels)
        
        # 1. 计算轮廓系数 (Silhouette Score) - 越大越好，范围: -1 到 1
        if len(self.data) < 50000:
            try:
                # 如果数据量较大，使用采样
                sample_size = min(10000, len(self.data))
                if len(self.data) > sample_size:
                    sample_indices = np.random.choice(len(self.data), sample_size, replace=False)
                    features_sample = features_array[sample_indices]
                    labels_sample = labels_array[sample_indices]
                else:
                    features_sample = features_array
                    labels_sample = labels_array
                
                metrics['silhouette_score'] = silhouette_score(
                    features_sample, 
                    labels_sample
                )
                logger.info(f"站点 {self.site_id} - Silhouette Score: {metrics['silhouette_score']:.4f}")
            except Exception as e:
                metrics['silhouette_score'] = None
                logger.warning(f"站点 {self.site_id} - Silhouette Score 计算失败: {str(e)}")
        else:
            metrics['silhouette_score'] = None
            logger.info(f"站点 {self.site_id} - 数据量过大（{len(self.data)}），跳过 Silhouette Score 计算")
        
        # 2. 计算 Calinski-Harabasz 指数 (CH指数) - 越大越好
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                features_array, 
                labels_array
            )
            logger.info(f"站点 {self.site_id} - Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
        except Exception as e:
            metrics['calinski_harabasz_score'] = None
            logger.warning(f"站点 {self.site_id} - Calinski-Harabasz Score 计算失败: {str(e)}")
        
        # 3. 计算 Davies-Bouldin 指数 (DB指数) - 越小越好
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(
                features_array, 
                labels_array
            )
            logger.info(f"站点 {self.site_id} - Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
        except Exception as e:
            metrics['davies_bouldin_score'] = None
            logger.warning(f"站点 {self.site_id} - Davies-Bouldin Score 计算失败: {str(e)}")
        
        # 如果提供了全局中心，计算与全局中心的差异
        if global_centers is not None:
            # 使用本地数据评估全局中心的性能
            from scipy.spatial.distance import cdist
            distances = cdist(features_array, global_centers, metric='euclidean')
            global_labels = np.argmin(distances, axis=1)
            
            # 计算全局中心在本地数据上的惯性
            global_inertia = np.sum(np.min(distances, axis=1) ** 2)
            metrics['global_inertia'] = global_inertia
            
            # 计算全局中心的评估指标
            try:
                metrics['global_silhouette_score'] = silhouette_score(
                    features_array, 
                    global_labels,
                    sample_size=min(10000, len(self.data)) if len(self.data) < 50000 else None
                ) if len(self.data) < 50000 else None
            except:
                metrics['global_silhouette_score'] = None
            
            try:
                metrics['global_calinski_harabasz_score'] = calinski_harabasz_score(
                    features_array, 
                    global_labels
                )
            except:
                metrics['global_calinski_harabasz_score'] = None
            
            try:
                metrics['global_davies_bouldin_score'] = davies_bouldin_score(
                    features_array, 
                    global_labels
                )
            except:
                metrics['global_davies_bouldin_score'] = None
            
            logger.info(f"站点 {self.site_id} 评估结果:")
            logger.info(f"  - 本地惯性: {metrics['inertia']:.2f}")
            logger.info(f"  - 全局惯性: {global_inertia:.2f}")
            if metrics.get('silhouette_score') is not None:
                logger.info(f"  - 本地 Silhouette: {metrics['silhouette_score']:.4f}")
                if metrics.get('global_silhouette_score') is not None:
                    logger.info(f"  - 全局 Silhouette: {metrics['global_silhouette_score']:.4f}")
            if metrics.get('calinski_harabasz_score') is not None:
                logger.info(f"  - 本地 CH指数: {metrics['calinski_harabasz_score']:.4f}")
                if metrics.get('global_calinski_harabasz_score') is not None:
                    logger.info(f"  - 全局 CH指数: {metrics['global_calinski_harabasz_score']:.4f}")
            if metrics.get('davies_bouldin_score') is not None:
                logger.info(f"  - 本地 DB指数: {metrics['davies_bouldin_score']:.4f}")
                if metrics.get('global_davies_bouldin_score') is not None:
                    logger.info(f"  - 全局 DB指数: {metrics['global_davies_bouldin_score']:.4f}")
        else:
            # 打印本地聚类评估结果
            logger.info(f"站点 {self.site_id} 聚类评估结果:")
            logger.info(f"  - 聚类数量: {metrics['n_clusters']}")
            logger.info(f"  - 样本数量: {metrics['n_samples']}")
            logger.info(f"  - 惯性: {metrics['inertia']:.2f}")
            if metrics.get('silhouette_score') is not None:
                logger.info(f"  - Silhouette Score: {metrics['silhouette_score']:.4f}")
            if metrics.get('calinski_harabasz_score') is not None:
                logger.info(f"  - Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
            if metrics.get('davies_bouldin_score') is not None:
                logger.info(f"  - Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
        
        return metrics


def run_local_clustering_all_sites(
    data_paths: Dict[int, str],
    n_clusters: int = 750,
    feature_columns: List[str] = None,
    random_state: int = 42,
    use_first_bloc_only: bool = True,
    auto_determine_k: bool = True,
    k_method: str = 'improved_heuristic',
    min_k: int = 2,
    max_k: int = 50,
    test_k_range: tuple = None
) -> Tuple[List[LocalSite], List[Dict]]:
    """
    在所有站点上运行本地聚类
    
    Parameters:
    -----------
    data_paths : Dict[int, str]
        站点ID到数据路径的映射
    n_clusters : int
        聚类数量（如果 auto_determine_k=False）
    feature_columns : List[str], optional
        特征列
    random_state : int
        随机种子
    use_first_bloc_only : bool, default=True
        是否只使用每个患者的第一条bloc数据
    auto_determine_k : bool, default=True
        是否自动为每个站点确定聚类数量
    k_method : str, default='improved_heuristic'
        确定聚类数量的方法：'improved_heuristic', 'elbow', 'silhouette'
    min_k : int, default=2
        最小聚类数量
    max_k : int, default=50
        最大聚类数量（建议根据数据规模调整）
    test_k_range : tuple, optional
        测试的k值范围（仅用于 elbow 和 silhouette 方法）
        
    Returns:
    --------
    Tuple[List[LocalSite], List[Dict]] : 站点列表和上传信息列表
    """
    logger.info("="*60)
    logger.info("开始在所有站点执行本地聚类")
    if auto_determine_k:
        logger.info(f"聚类策略: 自动为每个站点确定聚类数量 (方法: {k_method}, 范围: {min_k}-{max_k})")
    else:
        logger.info(f"聚类策略: 使用固定聚类数量 K={n_clusters}")
    logger.info("="*60)
    
    sites = []
    upload_infos = []
    
    for site_id, data_path in data_paths.items():
        logger.info(f"\n处理站点 {site_id}...")
        
        # 创建站点实例
        site = LocalSite(site_id, data_path, n_clusters)
        
        # 加载和预处理数据
        site.load_data(feature_columns, use_first_bloc_only=use_first_bloc_only)
        site.normalize_data()
        
        # 执行本地聚类
        clustering_result = site.perform_local_clustering(
            random_state, 
            auto_determine_k=auto_determine_k,
            k_method=k_method,
            min_k=min_k,
            max_k=max_k,
            test_k_range=test_k_range
        )
        
        # 评估聚类质量（计算 Silhouette、CH、DB 指标）
        logger.info(f"\n正在评估站点 {site_id} 的聚类质量...")
        evaluation_metrics = site.evaluate_clustering()
        
        # 获取上传信息
        upload_info = site.get_upload_info()
        # 将评估指标添加到上传信息中
        upload_info['evaluation_metrics'] = evaluation_metrics
        
        sites.append(site)
        upload_infos.append(upload_info)
        
        logger.info(f"站点 {site_id} 处理完成\n")
    
    logger.info("="*60)
    logger.info(f"所有 {len(sites)} 个站点的本地聚类已完成")
    logger.info("="*60)
    
    return sites, upload_infos


def visualize_clustering_results(
    sites: List[LocalSite],
    use_pca: bool = True,
    use_tsne: bool = False,
    max_samples_for_tsne: int = 5000,
    save_path: str = None
):
    """
    可视化所有站点的聚类结果
    
    Parameters:
    -----------
    sites : List[LocalSite]
        站点列表
    use_pca : bool, default=True
        是否使用PCA进行2D可视化
    use_tsne : bool, default=False
        是否使用t-SNE进行2D可视化（较慢但效果更好）
    max_samples_for_tsne : int, default=5000
        如果使用t-SNE，最大采样数量（避免计算时间过长）
    save_path : str, optional
        保存图片的路径，如果为None则显示图片
    """
    n_sites = len(sites)
    fig, axes = plt.subplots(2, n_sites, figsize=(6*n_sites, 12))
    
    if n_sites == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, site in enumerate(sites):
        site_id = site.site_id
        features = site.features.values
        labels = site.labels
        
        logger.info(f"正在为站点 {site_id} 准备可视化数据...")
        
        # 降维到2D
        if use_tsne and len(features) <= max_samples_for_tsne:
            # 使用t-SNE
            logger.info(f"  使用t-SNE降维（{len(features)}个样本）...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
            embedding = reducer.fit_transform(features)
            method_name = "t-SNE"
        else:
            # 使用PCA
            if use_tsne:
                logger.info(f"  样本数过多（{len(features)}），使用PCA代替t-SNE...")
            else:
                logger.info(f"  使用PCA降维...")
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(features)
            method_name = "PCA"
        
        # 第一行：2D聚类可视化
        ax1 = axes[0, idx]
        scatter = ax1.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=labels, 
            cmap='tab20', 
            alpha=0.6, 
            s=10
        )
        ax1.set_title(f'站点 {site_id} 聚类结果 ({method_name})\n'
                     f'样本数: {len(features)}, 簇数: {site.n_clusters}', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel(f'{method_name} 1')
        ax1.set_ylabel(f'{method_name} 2')
        ax1.grid(True, alpha=0.3)
        
        # 第二行：簇大小分布
        ax2 = axes[1, idx]
        cluster_sizes = site.cluster_sizes
        non_empty_clusters = cluster_sizes[cluster_sizes > 0]
        
        ax2.hist(non_empty_clusters, bins=min(50, len(non_empty_clusters)), 
                edgecolor='black', alpha=0.7, color='skyblue')
        ax2.set_title(f'站点 {site_id} 簇大小分布\n'
                     f'非空簇: {len(non_empty_clusters)}/{site.n_clusters}\n'
                     f'平均大小: {np.mean(non_empty_clusters):.1f}', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('簇大小')
        ax2.set_ylabel('频数')
        ax2.grid(True, alpha=0.3)
        
        logger.info(f"站点 {site_id} 可视化准备完成")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_cluster_statistics(sites: List[LocalSite], save_path: str = None):
    """
    可视化所有站点的聚类统计信息
    
    Parameters:
    -----------
    sites : List[LocalSite]
        站点列表
    save_path : str, optional
        保存图片的路径，如果为None则显示图片
    """
    n_sites = len(sites)
    
    # 准备数据
    site_ids = [site.site_id for site in sites]
    n_clusters = [site.n_clusters for site in sites]
    n_samples = [len(site.data) for site in sites]
    n_non_empty = [np.sum(site.cluster_sizes > 0) for site in sites]
    inertias = [site.local_model.inertia_ for site in sites]
    avg_cluster_sizes = [np.mean(site.cluster_sizes[site.cluster_sizes > 0]) 
                         for site in sites]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 簇数量对比
    ax1 = axes[0, 0]
    x_pos = np.arange(len(site_ids))
    ax1.bar(x_pos, n_clusters, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'站点 {sid}' for sid in site_ids])
    ax1.set_ylabel('簇数量', fontsize=11)
    ax1.set_title('各站点簇数量', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (nc, nne) in enumerate(zip(n_clusters, n_non_empty)):
        ax1.text(i, nc, f'{nne}/{nc}', ha='center', va='bottom', fontsize=9)
    
    # 2. 样本数量对比
    ax2 = axes[0, 1]
    ax2.bar(x_pos, n_samples, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'站点 {sid}' for sid in site_ids])
    ax2.set_ylabel('样本数量', fontsize=11)
    ax2.set_title('各站点样本数量', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, ns in enumerate(n_samples):
        ax2.text(i, ns, str(ns), ha='center', va='bottom', fontsize=9)
    
    # 3. 聚类惯性对比
    ax3 = axes[1, 0]
    ax3.bar(x_pos, inertias, alpha=0.7, color='mediumseagreen', edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'站点 {sid}' for sid in site_ids])
    ax3.set_ylabel('惯性值 (Inertia)', fontsize=11)
    ax3.set_title('各站点聚类惯性（越小越好）', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, inv in enumerate(inertias):
        ax3.text(i, inv, f'{inv:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 4. 平均簇大小对比
    ax4 = axes[1, 1]
    ax4.bar(x_pos, avg_cluster_sizes, alpha=0.7, color='plum', edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'站点 {sid}' for sid in site_ids])
    ax4.set_ylabel('平均簇大小', fontsize=11)
    ax4.set_title('各站点平均簇大小', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, acs in enumerate(avg_cluster_sizes):
        ax4.text(i, acs, f'{acs:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('本地聚类统计信息总览', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"统计信息图已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_evaluation_metrics(sites: List[LocalSite], save_path: str = None):
    """
    可视化所有站点的聚类评估指标
    
    Parameters:
    -----------
    sites : List[LocalSite]
        站点列表
    save_path : str, optional
        保存图片的路径，如果为None则显示图片
    """
    n_sites = len(sites)
    
    # 收集评估指标
    site_ids = [site.site_id for site in sites]
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    inertias = []
    
    for site in sites:
        metrics = site.evaluate_clustering()
        silhouette_scores.append(metrics.get('silhouette_score'))
        ch_scores.append(metrics.get('calinski_harabasz_score'))
        db_scores.append(metrics.get('davies_bouldin_score'))
        inertias.append(metrics.get('inertia'))
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Silhouette Score (轮廓系数)
    ax1 = axes[0, 0]
    valid_sil = [s for s in silhouette_scores if s is not None]
    valid_sil_ids = [site_ids[i] for i, s in enumerate(silhouette_scores) if s is not None]
    if valid_sil:
        x_pos = np.arange(len(valid_sil_ids))
        ax1.bar(x_pos, valid_sil, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'站点 {sid}' for sid in valid_sil_ids])
        ax1.set_ylabel('Silhouette Score', fontsize=11)
        ax1.set_title('轮廓系数 (Silhouette Score)\n越大越好 (范围: -1 到 1)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        for i, val in enumerate(valid_sil):
            ax1.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, '数据量过大\n无法计算', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('轮廓系数 (Silhouette Score)', fontsize=12, fontweight='bold')
    
    # 2. Calinski-Harabasz Score (CH指数)
    ax2 = axes[0, 1]
    valid_ch = [s for s in ch_scores if s is not None]
    valid_ch_ids = [site_ids[i] for i, s in enumerate(ch_scores) if s is not None]
    if valid_ch:
        x_pos = np.arange(len(valid_ch_ids))
        ax2.bar(x_pos, valid_ch, alpha=0.7, color='forestgreen', edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'站点 {sid}' for sid in valid_ch_ids])
        ax2.set_ylabel('Calinski-Harabasz Score', fontsize=11)
        ax2.set_title('Calinski-Harabasz 指数\n越大越好', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, val in enumerate(valid_ch):
            ax2.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, '计算失败', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Calinski-Harabasz 指数', fontsize=12, fontweight='bold')
    
    # 3. Davies-Bouldin Score (DB指数)
    ax3 = axes[1, 0]
    valid_db = [s for s in db_scores if s is not None]
    valid_db_ids = [site_ids[i] for i, s in enumerate(db_scores) if s is not None]
    if valid_db:
        x_pos = np.arange(len(valid_db_ids))
        ax3.bar(x_pos, valid_db, alpha=0.7, color='coral', edgecolor='black')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'站点 {sid}' for sid in valid_db_ids])
        ax3.set_ylabel('Davies-Bouldin Score', fontsize=11)
        ax3.set_title('Davies-Bouldin 指数\n越小越好', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for i, val in enumerate(valid_db):
            ax3.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, '计算失败', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Davies-Bouldin 指数', fontsize=12, fontweight='bold')
    
    # 4. Inertia (惯性)
    ax4 = axes[1, 1]
    valid_inertias = [i for i in inertias if i is not None]
    valid_inertia_ids = [site_ids[i] for i, inv in enumerate(inertias) if inv is not None]
    if valid_inertias:
        x_pos = np.arange(len(valid_inertia_ids))
        ax4.bar(x_pos, valid_inertias, alpha=0.7, color='mediumpurple', edgecolor='black')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'站点 {sid}' for sid in valid_inertia_ids])
        ax4.set_ylabel('Inertia', fontsize=11)
        ax4.set_title('聚类惯性 (Inertia)\n越小越好', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        for i, val in enumerate(valid_inertias):
            ax4.text(i, val, f'{val:.0f}', ha='center', va='bottom', fontsize=9, rotation=90)
    else:
        ax4.text(0.5, 0.5, '无数据', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('聚类惯性 (Inertia)', fontsize=12, fontweight='bold')
    
    plt.suptitle('聚类评估指标对比', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"评估指标可视化已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # 测试代码
    # 获取项目根目录（假设脚本在 fedkmeans/psowithkmeans/ 目录下）
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    data_paths = {
        1: os.path.join(project_root, "Dataset/datawithnum/group1_mimic_data.csv"),
        2: os.path.join(project_root, "Dataset/datawithnum/group2_mimic_data.csv"),
        3: os.path.join(project_root, "Dataset/datawithnum/group3_mimic_data.csv")
    }
    
    sites, upload_infos = run_local_clustering_all_sites(
        data_paths=data_paths,
        n_clusters=750,
        random_state=42,
        auto_determine_k=True  # 启用自动确定聚类数量
    )
    
    print("\n本地聚类完成！")
    print("\n" + "="*80)
    print("聚类评估指标总结")
    print("="*80)
    
    for info in upload_infos:
        site_id = info['site_id']
        print(f"\n站点 {site_id}:")
        print(f"  - 样本数: {info['n_samples']}")
        print(f"  - 聚类中心数: {info['cluster_centers'].shape[0]}")
        
        # 显示评估指标
        if 'evaluation_metrics' in info:
            metrics = info['evaluation_metrics']
            if metrics.get('inertia') is not None:
                print(f"  - 惯性 (Inertia): {metrics['inertia']:.2f}")
            else:
                print(f"  - 惯性 (Inertia): N/A")
            
            if metrics.get('silhouette_score') is not None:
                print(f"  - Silhouette Score: {metrics['silhouette_score']:.4f} (越大越好, 范围: -1 到 1)")
            else:
                print(f"  - Silhouette Score: N/A (数据量过大或计算失败)")
            
            if metrics.get('calinski_harabasz_score') is not None:
                print(f"  - Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f} (越大越好)")
            else:
                print(f"  - Calinski-Harabasz Score: N/A")
            
            if metrics.get('davies_bouldin_score') is not None:
                print(f"  - Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f} (越小越好)")
            else:
                print(f"  - Davies-Bouldin Score: N/A")
        else:
            print(f"  - 评估指标: 未计算")
    
    print("\n" + "="*80)
    
    # 可视化结果
    print("\n正在生成可视化结果...")
    try:
        # 可视化聚类结果
        visualize_clustering_results(
            sites=sites,
            use_pca=True,
            use_tsne=False,  # 设为True可启用t-SNE（较慢但效果更好）
            save_path=None  # 设为路径字符串可保存图片，如 "clustering_results.png"
        )
        
        # 可视化统计信息
        visualize_cluster_statistics(
            sites=sites,
            save_path=None  # 设为路径字符串可保存图片，如 "cluster_statistics.png"
        )
        
        # 可视化评估指标
        visualize_evaluation_metrics(
            sites=sites,
            save_path=None  # 设为路径字符串可保存图片，如 "evaluation_metrics.png"
        )
    except Exception as e:
        logger.warning(f"可视化过程中出现错误: {e}")
        logger.info("可以继续执行后续步骤")

