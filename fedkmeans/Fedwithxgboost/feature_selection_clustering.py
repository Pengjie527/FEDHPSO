"""
特征选择和聚类集成模块

该脚本整合了feature_importance_xgboost.py的特征筛选功能和local_clustering.py的聚类功能，
实现了首先使用BFO算法筛选特征，然后将筛选后的特征用于聚类分析。
"""
import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入特征筛选模块
from feature_importance_xgboost import (
    bacterial_foraging_feature_selection,
    get_feature_names_from_data
)

# 导入聚类模块
from local_clustering import (
    LocalSite,
    run_local_clustering_all_sites,
    visualize_clustering_results,
    visualize_cluster_statistics,
    visualize_evaluation_metrics
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_for_feature_selection(data_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    为特征选择加载数据
    
    Parameters:
    -----------
    data_path : str
        数据文件路径
    target_column : str
        目标列名称
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        特征数据框和目标系列
    """
    try:
        logger.info(f"加载数据用于特征选择: {data_path}")
        df = pd.read_csv(data_path)
        
        # 确保目标列存在
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于数据中")
        
        # 分离特征和目标
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"数据加载完成，形状: X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"加载数据时出错: {str(e)}")
        raise

def select_features_for_site(site_id: int, data_path: str, target_column: str, 
                            n_iterations: int = 10, population_size: int = 5, 
                            step_size: float = 0.1) -> List[str]:
    """
    为特定站点执行特征选择
    
    Parameters:
    -----------
    site_id : int
        站点ID
    data_path : str
        数据文件路径
    target_column : str
        目标列名称
    n_iterations : int, default=10
        BFO算法迭代次数
    population_size : int, default=5
        BFO算法种群大小
    step_size : float, default=0.1
        BFO算法步长
    
    Returns:
    --------
    List[str]
        筛选后的特征名称列表
    """
    logger.info(f"站点 {site_id}: 开始特征选择")
    
    # 加载数据
    X, y = load_data_for_feature_selection(data_path, target_column)
    
    # 获取特征名称
    feature_names = list(X.columns)
    
    # 执行BFO特征选择
    logger.info(f"站点 {site_id}: 执行BFO特征选择算法...")
    best_position, best_score, selected_indices = bacterial_foraging_feature_selection(
        X=X.values,
        y=y.values,
        feature_names=feature_names,
        n_iterations=n_iterations,
        population_size=population_size,
        step_size=step_size,
        logger_instance=logger
    )
    
    # 获取筛选后的特征名称
    selected_features = [feature_names[i] for i in selected_indices]
    
    logger.info(f"站点 {site_id}: 特征选择完成")
    logger.info(f"站点 {site_id}: 原始特征数: {len(feature_names)}, 筛选后特征数: {len(selected_features)}")
    logger.info(f"站点 {site_id}: 筛选后的特征: {', '.join(selected_features)}")
    
    return selected_features

def run_feature_selection_and_clustering(
    data_paths: Dict[int, str],
    target_column: str,
    n_clusters: int = 750,
    feature_selection_params: Optional[Dict] = None,
    clustering_params: Optional[Dict] = None,
    visualize_results: bool = True,
    use_common_features: bool = False
) -> Tuple[List[LocalSite], List[Dict], Dict[int, List[str]]]:
    """
    执行特征选择和聚类的集成流程
    
    Parameters:
    -----------
    data_paths : Dict[int, str]
        站点ID到数据路径的映射
    target_column : str
        目标列名称
    n_clusters : int, default=750
        聚类数量
    feature_selection_params : Dict, optional
        特征选择参数
    clustering_params : Dict, optional
        聚类参数
    visualize_results : bool, default=True
        是否可视化结果
    use_common_features : bool, default=False
        是否使用所有站点的共同特征
    
    Returns:
    --------
    Tuple[List[LocalSite], List[Dict], Dict[int, List[str]]]
        站点列表、上传信息列表和每个站点的特征选择结果
    """
    logger.info("="*80)
    logger.info("开始特征选择和聚类集成流程")
    logger.info("="*80)
    
    # 默认参数
    if feature_selection_params is None:
        feature_selection_params = {
            'n_iterations': 10,
            'population_size': 5,
            'step_size': 0.1
        }
    
    if clustering_params is None:
        clustering_params = {
            'random_state': 42,
            'use_first_bloc_only': True,
            'auto_determine_k': True,
            'k_method': 'improved_heuristic',
            'min_k': 2,
            'max_k': 50
        }
    
    # 1. 为每个站点执行特征选择
    site_features = {}
    for site_id, data_path in data_paths.items():
        try:
            site_features[site_id] = select_features_for_site(
                site_id=site_id,
                data_path=data_path,
                target_column=target_column,
                **feature_selection_params
            )
        except Exception as e:
            logger.error(f"站点 {site_id} 特征选择失败: {str(e)}")
            raise
    
    # 2. 如果需要，计算共同特征
    if use_common_features:
        all_features = list(site_features.values())
        common_features = list(set(all_features[0]).intersection(*all_features[1:]))
        logger.info(f"计算共同特征: 共 {len(common_features)} 个特征")
        logger.info(f"共同特征: {', '.join(common_features)}")
        # 更新每个站点的特征为共同特征
        for site_id in site_features:
            site_features[site_id] = common_features
    
    # 3. 为每个站点执行聚类，使用筛选后的特征
    sites = []
    upload_infos = []
    
    for site_id, data_path in data_paths.items():
        logger.info(f"\n处理站点 {site_id}...")
        logger.info(f"使用 {len(site_features[site_id])} 个筛选后的特征进行聚类")
        
        # 创建站点实例
        site = LocalSite(site_id, data_path, n_clusters)
        
        # 加载和预处理数据，使用筛选后的特征
        site.load_data(feature_columns=site_features[site_id], 
                      use_first_bloc_only=clustering_params['use_first_bloc_only'])
        site.normalize_data()
        
        # 执行本地聚类
        clustering_result = site.perform_local_clustering(
            random_state=clustering_params['random_state'],
            auto_determine_k=clustering_params['auto_determine_k'],
            k_method=clustering_params['k_method'],
            min_k=clustering_params['min_k'],
            max_k=clustering_params['max_k'],
            test_k_range=clustering_params.get('test_k_range')
        )
        
        # 评估聚类质量
        logger.info(f"\n正在评估站点 {site_id} 的聚类质量...")
        evaluation_metrics = site.evaluate_clustering()
        
        # 获取上传信息
        upload_info = site.get_upload_info()
        upload_info['evaluation_metrics'] = evaluation_metrics
        upload_info['selected_features'] = site_features[site_id]
        upload_info['n_selected_features'] = len(site_features[site_id])
        
        sites.append(site)
        upload_infos.append(upload_info)
        
        logger.info(f"站点 {site_id} 处理完成")
    
    # 4. 可视化结果
    if visualize_results:
        try:
            logger.info("\n开始生成可视化结果...")
            
            # 可视化聚类结果
            visualize_clustering_results(
                sites=sites,
                use_pca=True,
                use_tsne=False
            )
            
            # 可视化统计信息
            visualize_cluster_statistics(
                sites=sites
            )
            
            # 可视化评估指标
            visualize_evaluation_metrics(
                sites=sites
            )
            
            logger.info("可视化完成")
        except Exception as e:
            logger.warning(f"可视化过程中出现错误: {str(e)}")
            logger.info("可以继续执行后续步骤")
    
    logger.info("="*80)
    logger.info(f"特征选择和聚类集成流程完成，处理了 {len(sites)} 个站点")
    logger.info("="*80)
    
    return sites, upload_infos, site_features

def save_feature_selection_results(site_features: Dict[int, List[str]], output_path: str):
    """
    保存特征选择结果
    
    Parameters:
    -----------
    site_features : Dict[int, List[str]]
        站点ID到筛选特征列表的映射
    output_path : str
        输出文件路径
    """
    # 创建结果字典
    results = []
    for site_id, features in site_features.items():
        results.append({
            'site_id': site_id,
            'selected_features': features,
            'n_features': len(features)
        })
    
    # 保存为JSON文件
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"特征选择结果已保存至: {output_path}")

if __name__ == "__main__":
    # 测试代码
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # 定义数据路径
    data_paths = {
        1: os.path.join(project_root, "Dataset/datawithser/group1_mimic_data.csv"),
        2: os.path.join(project_root, "Dataset/datawithser/group2_mimic_data.csv"),
        3: os.path.join(project_root, "Dataset/datawithser/group3_mimic_data.csv")
    }
    
    # 目标列（需要根据实际数据修改）
    # 注意：这里假设数据中有一个目标列用于特征选择
    # 如果是无监督任务，可能需要修改特征选择部分的代码
    target_column = "sepsis_label"  # 示例目标列，需要根据实际数据修改
    
    # 运行特征选择和聚类
    sites, upload_infos, site_features = run_feature_selection_and_clustering(
        data_paths=data_paths,
        target_column=target_column,
        n_clusters=750,
        feature_selection_params={
            'n_iterations': 5,  # 减少迭代次数以加快演示
            'population_size': 5,
            'step_size': 0.1
        },
        clustering_params={
            'random_state': 42,
            'use_first_bloc_only': True,
            'auto_determine_k': True,
            'k_method': 'improved_heuristic'
        },
        visualize_results=True,
        use_common_features=False
    )
    
    # 保存特征选择结果
    output_path = os.path.join(project_root, "feature_selection_results.json")
    save_feature_selection_results(site_features, output_path)
    
    # 打印总结信息
    print("\n特征选择和聚类总结")
    print("="*60)
    
    for info in upload_infos:
        site_id = info['site_id']
        print(f"\n站点 {site_id}:")
        print(f"  - 样本数: {info['n_samples']}")
        print(f"  - 筛选后特征数: {info['n_selected_features']}")
        print(f"  - 聚类中心数: {info['cluster_centers'].shape[0]}")
        
        # 显示评估指标
        if 'evaluation_metrics' in info:
            metrics = info['evaluation_metrics']
            if metrics.get('inertia') is not None:
                print(f"  - 惯性 (Inertia): {metrics['inertia']:.2f}")
            
            if metrics.get('silhouette_score') is not None:
                print(f"  - Silhouette Score: {metrics['silhouette_score']:.4f}")
            
            if metrics.get('calinski_harabasz_score') is not None:
                print(f"  - Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
            
            if metrics.get('davies_bouldin_score') is not None:
                print(f"  - Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
    
    print("\n" + "="*60)