# supervised_clustering_pipeline.py

import numpy as np
import pandas as pd
import logging
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import List, Dict, Tuple

# 确保你的预处理模块路径在 sys.path 中
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

import shap
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_top_k_features_by_xgboost(
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 10,
        random_state: int = 42
) -> List[str]:
    """
    基于 XGBoost + SHAP 提取 top_k 特征
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.Series(mean_shap, index=X.columns).sort_values(ascending=False)
    top_features = list(shap_importance.head(top_k).index)
    logger.info(f"Top {top_k} 特征（按 SHAP 排序）: {top_features}")
    return top_features


class LocalSiteSupervised:
    """
    本地站点类 — 加入监督特征选择 + 聚类流程
    """

    def __init__(self, site_id: int, data_path: str, label_name: str = 'mortality',
                 top_k: int = 10, n_clusters: int = 5):
        self.site_id = site_id
        self.data_path = data_path
        self.label_name = label_name
        self.top_k = top_k
        self.n_clusters = n_clusters

        self.data = None
        self.features = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.selected_features = None

        self.local_model = None
        self.labels = None

        logger.info(f"站点 {site_id} 初始化完成 (监督聚类模式)")

    def load_and_preprocess(self, use_first_bloc_only: bool = True):
        """
        加载并预处理数据
        """
        config = PreprocessingConfig(
            use_first_bloc_only=use_first_bloc_only,
            feature_columns=None  # 默认使用所有预定义特征
        )
        self.data, features_all, self.feature_columns, self.scaler = preprocess_data(
            data_path=self.data_path,
            config=config,
            site_id=self.site_id,
            logger_instance=logger
        )
        self.features = features_all.copy()
        return self.features

    def supervised_feature_selection(self):
        """
        基于 label 提取 top_k 特征
        """
        if self.label_name not in self.data.columns:
            raise RuntimeError(f"标签 {self.label_name} 在数据中不存在")
        y = self.data[self.label_name]
        X = self.features.copy()
        top_feats = select_top_k_features_by_xgboost(X, y, top_k=self.top_k)
        self.selected_features = top_feats
        # 更新 features，仅保留 top_k
        self.features = X[top_feats]
        logger.info(f"站点 {self.site_id}: 使用特征 {self.selected_features} 进行聚类")

    def fit_kmeans(self, random_state: int = 42):
        """
        使用 K‑Means 对 selected_features 进行聚类
        """
        self.local_model = KMeans(n_clusters=self.n_clusters, random_state=random_state, n_init=10, max_iter=300)
        self.labels = self.local_model.fit_predict(self.features)
        logger.info(f"站点 {self.site_id}: KMeans 聚类完成 (K={self.n_clusters})")
        return self.labels

    def evaluate(self) -> Dict:
        """
        评估聚类效果
        """
        metrics = {'site_id': self.site_id, 'n_clusters': self.n_clusters}
        X_arr = self.features.values
        labels = self.labels

        try:
            metrics['silhouette_score'] = silhouette_score(X_arr, labels)
        except:
            metrics['silhouette_score'] = None
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_arr, labels)
        except:
            metrics['calinski_harabasz_score'] = None
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_arr, labels)
        except:
            metrics['davies_bouldin_score'] = None

        logger.info(f"站点 {self.site_id} 聚类评估: {metrics}")
        return metrics


def run_supervised_pipeline(
        data_paths: Dict[int, str],
        label_name: str = 'mortality',
        top_k: int = 10,
        n_clusters: int = 5,
        random_state: int = 42,
        use_first_bloc_only: bool = True
) -> Tuple[List[LocalSiteSupervised], List[Dict]]:
    """
    在多个站点上运行监督特征选择 + 聚类流程
    """
    sites = []
    metrics_list = []

    for site_id, path in data_paths.items():
        logger.info(f"=== 开始站点 {site_id} ===")
        site = LocalSiteSupervised(site_id, path, label_name=label_name, top_k=top_k, n_clusters=n_clusters)
        site.load_and_preprocess(use_first_bloc_only=use_first_bloc_only)
        site.supervised_feature_selection()
        site.fit_kmeans(random_state=random_state)
        metrics = site.evaluate()
        sites.append(site)
        metrics_list.append(metrics)
        logger.info(f"=== 结束站点 {site_id} ===\n")

    return sites, metrics_list


if __name__ == "__main__":
    # 示例使用方式
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_paths = {
        1: os.path.join(project_root, "Dataset/datawithser/group1_mimic_data.csv"),
        2: os.path.join(project_root, "Dataset/datawithser/group2_mimic_data.csv"),
        3: os.path.join(project_root, "Dataset/datawithser/group3_mimic_data.csv")
    }
    sites, metrics = run_supervised_pipeline(
        data_paths=data_paths,
        label_name='mortality',  # 或 'SOFA'
        top_k=10,
        n_clusters=5,
        random_state=42,
        use_first_bloc_only=True
    )

    print("监督聚类流程完成。评估结果：")
    for m in metrics:
        print(m)
