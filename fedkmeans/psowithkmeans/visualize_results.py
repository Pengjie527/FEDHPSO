"""
visualize_results.py
用于 FedH-PSO 结果可视化
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_results(output_path: str):
    """
    可视化 FedH-PSO 实验结果：
    - 全局簇中心
    - 局部簇中心
    - 收敛曲线
    - 站点权重
    """

    # ==========================
    # 1. 加载全局结果
    # ==========================
    global_centers = np.load(os.path.join(output_path, "global_centers.npy"))

    with open(os.path.join(output_path, "weights.json"), 'r') as f:
        weights_data = json.load(f)
    weights = np.array(weights_data['weights'])
    site_ids = weights_data['site_ids']

    with open(os.path.join(output_path, "convergence_history.json"), 'r') as f:
        convergence_data = json.load(f)
    convergence_history = convergence_data['convergence_history']
    fitness_history = convergence_data['fitness_history']

    # ==========================
    # 2. 绘制收敛曲线
    # ==========================
    plt.figure(figsize=(10, 4))
    plt.plot(convergence_history, marker='o')
    plt.title("Global Center Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Center Difference (Frobenius norm)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "convergence_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(fitness_history, marker='o', color='orange')
    plt.title("PSO Fitness Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "fitness_curve.png"))
    plt.close()

    # ==========================
    # 3. 绘制站点权重
    # ==========================
    plt.figure(figsize=(8, 4))
    plt.bar([str(i) for i in site_ids], weights)
    plt.title("Site Weights")
    plt.xlabel("Site ID")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "site_weights.png"))
    plt.close()

    # ==========================
    # 4. 全局簇 + 局部簇可视化 (降维)
    # ==========================
    # 尝试PCA降到2D
    all_local_centers = []
    try:
        with open(os.path.join(output_path, "summary.json"), 'r') as f:
            summary = json.load(f)
        # 读取局部簇
        for site_id in summary['site_info'].keys():
            local_centers_path = os.path.join(output_path, f"local_centers_site{site_id}.npy")
            if os.path.exists(local_centers_path):
                centers = np.load(local_centers_path)
                all_local_centers.append(centers)
    except:
        pass

    if all_local_centers:
        all_local_centers = np.vstack(all_local_centers)
        pca = PCA(n_components=2)
        all_points_2d = pca.fit_transform(np.vstack([all_local_centers, global_centers]))
        local_points_2d = all_points_2d[:all_local_centers.shape[0], :]
        global_points_2d = all_points_2d[all_local_centers.shape[0]:, :]

        plt.figure(figsize=(8, 6))
        plt.scatter(local_points_2d[:, 0], local_points_2d[:, 1], s=30, c='lightblue', label='Local Centers', alpha=0.6)
        plt.scatter(global_points_2d[:, 0], global_points_2d[:, 1], s=100, c='red', marker='X', label='Global Centers')
        plt.title("Local Centers vs Global Centers (PCA 2D)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "global_vs_local.png"))
        plt.close()

    print(f"✓ 可视化完成，文件保存至 {output_path}")
