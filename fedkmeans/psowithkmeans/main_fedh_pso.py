"""
FedH-PSO 主程序入口
运行基于PSO优化的联邦层次化聚类
"""
import os
import numpy as np
from datetime import datetime
import logging
import json
from local_clustering import LocalSite
from fedh_pso import FedHPSO
from visualize_results import visualize_results

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 修复 numpy 兼容性问题
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# ==========================
# 配置参数区
# ==========================
K_LOCAL = 76             # 每个站点局部聚类簇数
K_GLOBAL = 50           # 全局聚类簇数
PSO_PARTICLES = 10
PSO_MAX_ITER = 50
PSO_OMEGA = 0.3
PSO_C1 = 0.5
PSO_C2 = 0.1
MAX_GLOBAL_ITER = 30
CONVERGENCE_THRESHOLD = 1e-4

SITE_DATA_PATHS = {
    1: "E:/研三/py_ai_clinician-master/Dataset/datawithnum/group1_mimic_data.csv",
    2: "E:/研三/py_ai_clinician-master/Dataset/datawithnum/group2_mimic_data.csv",
    3: "E:/研三/py_ai_clinician-master/Dataset/datawithnum/group3_mimic_data.csv"
}

OUTPUT_DIR = "results_fedh_pso"


# ==========================
# 主实验函数
# ==========================
def run_fedh_pso_experiment(site_data_paths: dict,
                            k_local: int = K_LOCAL,
                            k_global: int = K_GLOBAL,
                            output_dir: str = OUTPUT_DIR) -> dict:
    """
    运行完整的 FedH-PSO 实验，自动保存结果并生成可视化
    """
    logger.info("\n" + "=" * 100)
    logger.info("FedH-PSO: 联邦层次化聚类实验")
    logger.info("=" * 100)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"fedh_pso_{timestamp}")
    os.makedirs(output_path, exist_ok=True)

    # ==========================
    # 阶段 1: 本地聚类
    # ==========================
    logger.info("\n>>> 阶段 1: 本地站点执行 K-Means 聚类")
    local_results = {}
    for site_id, data_path in site_data_paths.items():
        logger.info(f"\n--- 站点 {site_id} ---")
        site = LocalSite(site_id=site_id, data_path=data_path, n_clusters=k_local)
        site.load_data(use_first_bloc_only=True)
        site.normalize_data()
        site.perform_local_clustering(random_state=42)
        local_results[site_id] = {
            'cluster_centers': site.cluster_centers,
            'labels': site.labels,
            'inertia': site.local_model.inertia_,
            'n_samples': len(site.labels),
            'site_id': site_id
        }
        logger.info(
            f"站点 {site_id} 本地聚类完成: 聚类中心 {site.cluster_centers.shape}, SSE {site.local_model.inertia_:.2f}")

    # ==========================
    # 阶段 2: 准备上传信息
    # ==========================
    logger.info("\n>>> 阶段 2: 准备上传信息")
    upload_infos = []
    for site_id, res in local_results.items():
        upload_infos.append({
            'site_id': site_id,
            'n_samples': res['n_samples'],
            'inertia': res['inertia'],
            'cluster_centers': res['cluster_centers']
        })
    n_features = local_results[list(local_results.keys())[0]]['cluster_centers'].shape[1]

    # ==========================
    # 阶段 3: FedH-PSO 联邦聚类
    # ==========================
    logger.info("\n>>> 阶段 3: FedH-PSO 联邦聚类优化")
    fedh_pso = FedHPSO(
        n_clusters=k_global,
        n_features=n_features,
        pso_omega=PSO_OMEGA,
        pso_c1=PSO_C1,
        pso_c2=PSO_C2,
        pso_max_iter=PSO_MAX_ITER,
        convergence_threshold=CONVERGENCE_THRESHOLD
    )
    fedh_results = fedh_pso.federated_clustering(
        upload_infos=upload_infos,
        max_global_iter=MAX_GLOBAL_ITER
    )

    # ==========================
    # 阶段 4: 保存结果
    # ==========================
    save_fedh_results(output_path, fedh_results, upload_infos, local_results)

    # ==========================
    # 阶段 5: 可视化
    # ==========================
    visualize_results(output_path)

    return {
        'local_results': local_results,
        'fedh_results': fedh_results,
        'output_path': output_path
    }


# ==========================
# 保存结果函数
# ==========================
def save_fedh_results(output_path: str,
                      fedh_results: dict,
                      upload_infos: list,
                      local_results: dict):
    """保存 FedH-PSO 实验结果，包含全局中心、权重、收敛历史、机构聚类、汇总信息"""
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存全局聚类中心
    np.save(os.path.join(output_path, "global_centers.npy"), fedh_results['global_centers'])

    # 2. 保存权重
    weights_data = {
        'weights': fedh_results['weights'].tolist(),
        'site_ids': [info['site_id'] for info in upload_infos]
    }
    with open(os.path.join(output_path, "weights.json"), 'w') as f:
        json.dump(weights_data, f, indent=2)

    # 3. 保存收敛历史
    with open(os.path.join(output_path, "convergence_history.json"), 'w') as f:
        json.dump({
            'convergence_history': fedh_results['convergence_history'],
            'fitness_history': fedh_results['fitness_history']
        }, f, indent=2)

    # 4. 保存机构聚类结果
    similarity_matrix = fedh_results.get('similarity_matrix', [])
    with open(os.path.join(output_path, "institution_clusters.json"), 'w') as f:
        json.dump({
            'institution_labels': fedh_results['institution_labels'].tolist(),
            'similarity_matrix': similarity_matrix.tolist() if len(similarity_matrix) > 0 else []
        }, f, indent=2)

    # 5. 保存实验汇总
    summary = {
        'timestamp': timestamp,
        'n_clusters': fedh_results.get('global_centers', np.array([])).shape[0],
        'n_features': fedh_results.get('global_centers', np.array([])).shape[1]
        if fedh_results.get('global_centers', np.array([])).size > 0 else 0,
        'n_sites': len(upload_infos),
        'n_iterations': fedh_results['n_iterations'],
        'converged': fedh_results['converged'],
        'site_info': {
            info['site_id']: {
                'n_samples': local_results[info['site_id']]['n_samples'],
                'inertia': float(local_results[info['site_id']]['inertia'])
            } for info in upload_infos
        }
    }
    with open(os.path.join(output_path, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ FedH-PSO 结果已保存至 {output_path}")


# ==========================
# 主函数入口
# ==========================
def main():
    # 检查数据文件是否存在
    for site_id, path in SITE_DATA_PATHS.items():
        if not os.path.exists(path):
            logger.error(f"数据文件不存在: {path}")
            return

    # 运行实验
    results = run_fedh_pso_experiment(
        site_data_paths=SITE_DATA_PATHS,
        k_local=K_LOCAL,
        k_global=K_GLOBAL,
        output_dir=OUTPUT_DIR
    )

    logger.info("\n✅ FedH-PSO 实验完成!")
    logger.info(f"结果保存在: {results['output_path']}")


if __name__ == "__main__":
    main()



