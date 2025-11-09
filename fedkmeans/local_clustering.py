"""
Step 1: Local K-Means Clustering for Each Site
每个站点本地执行K-Means聚类
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import logging

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
        logger.info(f"站点 {self.site_id} 正在加载数据: {self.data_path}")
        df_raw = pd.read_csv(self.data_path)
        
        logger.info(f"  原始数据: {len(df_raw)} 条记录")
        
        # 如果需要只使用第一条bloc数据
        if use_first_bloc_only:
            if 'icustayid' in df_raw.columns and 'bloc' in df_raw.columns:
                # 找出每个患者的最小bloc值（第一个bloc）
                first_blocs = df_raw.groupby('icustayid')['bloc'].min().reset_index()
                first_blocs.columns = ['icustayid', 'first_bloc']
                
                # 合并并筛选第一个bloc的数据
                df_with_first = df_raw.merge(first_blocs, on='icustayid')
                self.data = df_with_first[df_with_first['bloc'] == df_with_first['first_bloc']].copy()
                
                # 删除辅助列
                if 'first_bloc' in self.data.columns:
                    self.data = self.data.drop('first_bloc', axis=1)
                
                # 处理可能的重复（同一患者的第一个bloc有多条记录的情况）
                if self.data.groupby('icustayid').size().max() > 1:
                    logger.warning(f"  发现部分患者在第一个bloc有多条记录，保留第一条")
                    self.data = self.data.groupby('icustayid').first().reset_index()
                
                n_patients = self.data['icustayid'].nunique()
                logger.info(f"  提取第一个bloc数据: {len(self.data)} 条记录 ({n_patients} 个患者)")
            else:
                logger.warning(f"  警告: 数据中缺少 'icustayid' 或 'bloc' 列，使用全部数据")
                self.data = df_raw.copy()
        else:
            self.data = df_raw.copy()
            logger.info(f"  使用全部数据: {len(self.data)} 条记录")
        
        if feature_columns is None:
            # 根据原始论文，只使用关键的生理学特征
            # 二进制特征 (0/1)
            binary_features = ['gender', 'mechvent', 're_admission']  # 去掉 max_dose_vaso
            
            # 正态分布特征 (标准化)
            normal_features = ['age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 
                            'RR', 'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 
                            'Glucose', 'Magnesium', 'Calcium', 'Ionised_Ca', 'CO2_mEqL', 
                            'Shock_Index', 'PaO2_FiO2']
            
            # 对数正态分布特征 (log(0.1+x)后标准化)
            lognormal_features = ['SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 
                                'INR', 'input_total', 'input_4hourly', 'output_total', 
                                'output_4hourly', 'Hb', 'WBC_count', 'Platelets_count', 'PTT', 
                                'PT', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3', 
                                'Arterial_lactate', 'Albumin', 'cumulated_balance']
            
            # 组合所有需要的特征
            important_features = binary_features + normal_features + lognormal_features
            
            # 从实际数据中选择存在的特征
            available_features = [col for col in important_features if col in self.data.columns]
            
            # 警告缺失特征
            missing_features = set(important_features) - set(available_features)
            if missing_features:
                logger.warning(f"  缺少以下特征: {list(missing_features)}")
            
            feature_columns = available_features
            
            # 如果没有重要特征，则回退到原来的方法
            if not feature_columns:
                logger.warning("  使用备用方法选择特征...")
                exclude_cols = ['bloc', 'icustayid', 'charttime', 'gender', 'age',
                                'died_in_hosp', 'died_within_48h_of_out_time', 
                                'mortality_90d', 'delay_end_of_record_and_discharge_or_death']
                feature_columns = [col for col in self.data.columns 
                                if col not in exclude_cols and 
                                self.data[col].dtype in ['float64', 'int64']]

        self.features = self.data[feature_columns].copy()

        # ---------------- 处理缺失值 ----------------
        # lognormal 特征用中位数填充，其余用均值
        for col in self.features.columns:
            if col in lognormal_features:
                self.features[col].fillna(self.features[col].median(), inplace=True)
            else:
                self.features[col].fillna(self.features[col].mean(), inplace=True)

        # ---------------- 清理极端异常值 ----------------
        for col in self.features.columns:
            q1 = self.features[col].quantile(0.01)
            q99 = self.features[col].quantile(0.99)
            self.features[col] = self.features[col].clip(lower=q1, upper=q99)

        # 替换 inf 值
        self.features.replace([np.inf, -np.inf], 0, inplace=True)
        self.features.fillna(0, inplace=True)

        # ---------------- 特征标准化 ----------------
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        to_scale = normal_features + lognormal_features
        to_scale = [col for col in to_scale if col in self.features.columns]
        if to_scale:
            self.features[to_scale] = scaler.fit_transform(self.features[to_scale])

        # ---------------- 二进制特征偏移 ----------------
        binary_cols = [col for col in binary_features if col in self.features.columns]
        if binary_cols:
            self.features[binary_cols] = self.features[binary_cols] - 0.5

        logger.info(f"站点 {self.site_id} 数据加载完成: {len(self.data)} 条记录, {len(feature_columns)} 个特征")
        logger.info(f"特征列: {feature_columns[:5]}... (共{len(feature_columns)}个)")

        return self.features
    
    def normalize_data(self):
        """标准化数据
        
        - 二进制特征: 已经 -0.5 处理过
        - 正态分布特征: Z-score标准化
        - 对数正态特征: log(0.1+x) 然后 Z-score标准化
        """
        logger.info(f"站点 {self.site_id} 正在标准化数据...")

        # 二进制特征
        binary_cols = ['gender', 'mechvent', 're_admission']
        binary_cols = [col for col in binary_cols if col in self.features.columns]

        # 正态分布特征
        normal_cols = ['age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 
                    'RR', 'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 
                    'Glucose', 'Magnesium', 'Calcium']
        normal_cols = [col for col in normal_cols if col in self.features.columns]

        # 对数正态分布特征
        lognormal_cols = ['SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 
                        'INR', 'input_total', 'input_4hourly', 'output_total', 'output_4hourly',
                        'Hb', 'WBC_count', 'Platelets_count', 'PTT', 'PT', 'Arterial_pH',
                        'paO2', 'paCO2', 'Arterial_BE', 'HCO3', 'Arterial_lactate']
        lognormal_cols = [col for col in lognormal_cols if col in self.features.columns]

        # 对数正态特征：log(0.1 + x)
        for col in lognormal_cols:
            self.features[col] = np.log(0.1 + self.features[col].clip(lower=0))
        
        # 标准化前填充缺失值
        for col in normal_cols + lognormal_cols:
            self.features[col].fillna(self.features[col].median(), inplace=True)

        # 标准化
        to_scale = normal_cols + lognormal_cols
        self.features[to_scale] = self.scaler.fit_transform(self.features[to_scale])

        return self.features

    
    def determine_optimal_clusters(self):
        n_samples = len(self.data)
        n_features = len(self.features.columns)

        # 核心启发式规则
        k_samples = max(1, int(np.sqrt(n_samples / 2)))
        k_features = max(1, n_features * 2)

        # 自动选择合理簇数
        k = int(np.median([k_samples, k_features]))
        return k


    
    def perform_local_clustering(self, random_state: int = 42, auto_determine_k: bool = True):
        """
        执行本地K-Means聚类
        
        Parameters:
        -----------
        random_state : int
            随机种子
        auto_determine_k : bool, default=True
            是否自动确定聚类数量
            
        Returns:
        --------
        Dict : 包含聚类结果的字典
        """
        # 自动确定聚类数量
        if auto_determine_k:
            optimal_k = self.determine_optimal_clusters()
            self.n_clusters = optimal_k
            logger.info(f"站点 {self.site_id} 使用自动确定的聚类数量: {self.n_clusters}")
        
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
        Dict : 评估指标
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        metrics = {
            'site_id': self.site_id,
            'inertia': self.local_model.inertia_,
        }
        
        # 计算轮廓系数 (如果数据量不是太大)
        if len(self.data) < 50000:
            try:
                metrics['silhouette_score'] = silhouette_score(
                    self.features, 
                    self.labels,
                    sample_size=min(10000, len(self.data))
                )
            except:
                metrics['silhouette_score'] = None
        else:
            metrics['silhouette_score'] = None
        
        # 计算 Davies-Bouldin 指数
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(
                self.features, 
                self.labels
            )
        except:
            metrics['davies_bouldin_score'] = None
        
        # 如果提供了全局中心，计算与全局中心的差异
        if global_centers is not None:
            # 使用本地数据评估全局中心的性能
            from scipy.spatial.distance import cdist
            distances = cdist(self.features, global_centers, metric='euclidean')
            global_labels = np.argmin(distances, axis=1)
            
            # 计算全局中心在本地数据上的惯性
            global_inertia = np.sum(np.min(distances, axis=1) ** 2)
            metrics['global_inertia'] = global_inertia
            
            logger.info(f"站点 {self.site_id} 评估结果:")
            logger.info(f"  - 本地惯性: {metrics['inertia']:.2f}")
            logger.info(f"  - 全局惯性: {global_inertia:.2f}")
        
        return metrics


def run_local_clustering_all_sites(
    data_paths: Dict[int, str],
    n_clusters: int = 750,
    feature_columns: List[str] = None,
    random_state: int = 42,
    use_first_bloc_only: bool = True,
    auto_determine_k: bool = True
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
        
    Returns:
    --------
    Tuple[List[LocalSite], List[Dict]] : 站点列表和上传信息列表
    """
    logger.info("="*60)
    logger.info("开始在所有站点执行本地聚类")
    if auto_determine_k:
        logger.info("聚类策略: 自动为每个站点确定聚类数量")
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
        clustering_result = site.perform_local_clustering(random_state, auto_determine_k=auto_determine_k)
        
        # 获取上传信息
        upload_info = site.get_upload_info()
        
        sites.append(site)
        upload_infos.append(upload_info)
        
        logger.info(f"站点 {site_id} 处理完成\n")
    
    logger.info("="*60)
    logger.info(f"所有 {len(sites)} 个站点的本地聚类已完成")
    logger.info("="*60)
    
    return sites, upload_infos


if __name__ == "__main__":
    # 测试代码
    data_paths = {
        1: "Dataset/dataave/group1_mimic_data.csv",
        2: "Dataset/dataave/group2_mimic_data.csv",
        3: "Dataset/dataave/group3_mimic_data.csv"
    }
    
    sites, upload_infos = run_local_clustering_all_sites(
        data_paths=data_paths,
        n_clusters=750,
        random_state=42,
        auto_determine_k=True  # 启用自动确定聚类数量
    )
    
    print("\n本地聚类完成！")
    for info in upload_infos:
        print(f"站点 {info['site_id']}: {info['n_samples']} 样本, "
              f"使用了 {info['cluster_centers'].shape[0]} 个聚类中心")

