import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PSOOptimizer:
    def __init__(self,
                 n_features: int,
                 n_clusters_global: int = None,
                 n_particles: int = 10,
                 max_iter: int = 100,
                 omega: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        初始化 PSO 优化器

        Parameters
        ----------
        n_features : int
            特征维度
        n_clusters_global : int, optional
            全局簇数量，如果 None 则自动取第一个站点的局部簇数量
        n_particles : int
            粒子数量
        max_iter : int
            最大迭代次数
        omega : float
            惯性权重
        c1 : float
            认知因子
        c2 : float
            社会因子
        """
        self.n_features = n_features
        self.n_clusters_global = n_clusters_global
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.particle_dim = None  # 在初始化粒子时自动计算

        logger.info(f"PSO初始化: n_features={n_features}, "
                    f"K_global={n_clusters_global}, "
                    f"n_particles={n_particles}, max_iter={max_iter}")

    def _initialize_particles(self, local_centers_list: List[np.ndarray]):
        if self.n_clusters_global is None:
            self.n_clusters_global = local_centers_list[0].shape[0]

        self.particle_dim = self.n_clusters_global * self.n_features

        particles = np.zeros((self.n_particles, self.particle_dim))
        velocities = np.zeros_like(particles)

        for i, centers in enumerate(local_centers_list):
            centers_flat = centers.flatten()

            # 自动兼容局部簇数不同
            if centers.shape[0] < self.n_clusters_global:
                pad = np.tile(centers.mean(axis=0),
                              (self.n_clusters_global - centers.shape[0], 1))
                centers_flat = np.concatenate([centers_flat, pad.flatten()])
            elif centers.shape[0] > self.n_clusters_global:
                centers_flat = centers[:self.n_clusters_global, :].flatten()

            idx = i % self.n_particles
            particles[idx] = centers_flat

        return particles, velocities

    def _reshape_particle(self, particle: np.ndarray) -> np.ndarray:
        return particle.reshape(self.n_clusters_global, self.n_features)

    def _compute_fitness(self, particle: np.ndarray,
                         local_centers_list: List[np.ndarray],
                         weights: List[float]) -> float:
        global_centers = self._reshape_particle(particle)
        fitness = 0.0

        for centers, weight in zip(local_centers_list, weights):
            n_local = centers.shape[0]
            n_global = self.n_clusters_global

            if n_local < n_global:
                pad = np.tile(centers.mean(axis=0), (n_global - n_local, 1))
                centers_exp = np.vstack([centers, pad])
            else:
                centers_exp = centers[:n_global, :]

            distances_sq = np.sum((global_centers - centers_exp) ** 2, axis=1)
            fitness += weight * np.sum(distances_sq)

        return fitness

    def optimize(self, local_centers_list: List[np.ndarray],
                 weights: List[float]) -> Tuple[np.ndarray, List[float]]:
        logger.info("\n" + "="*60)
        logger.info("开始 PSO 优化全局聚类中心")
        logger.info("="*60)

        particles, velocities = self._initialize_particles(local_centers_list)
        best_positions = particles.copy()
        best_fitnesses = np.array([
            self._compute_fitness(particles[i], local_centers_list, weights)
            for i in range(self.n_particles)
        ])

        global_best_idx = np.argmin(best_fitnesses)
        global_best_particle = particles[global_best_idx].copy()
        global_best_fitness = best_fitnesses[global_best_idx]

        logger.info(f"初始最佳适应度: {global_best_fitness:.6f}")

        fitness_history = []

        for iter_num in range(self.max_iter):
            for i in range(self.n_particles):
                r1 = np.random.random(self.particle_dim)
                r2 = np.random.random(self.particle_dim)
                cognitive = self.c1 * r1 * (best_positions[i] - particles[i])
                social = self.c2 * r2 * (global_best_particle - particles[i])
                velocities[i] = self.omega * velocities[i] + cognitive + social
                particles[i] += velocities[i]

                fitness = self._compute_fitness(particles[i], local_centers_list, weights)
                if fitness < best_fitnesses[i]:
                    best_fitnesses[i] = fitness
                    best_positions[i] = particles[i].copy()
                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_particle = particles[i].copy()

            fitness_history.append(global_best_fitness)
            if (iter_num + 1) % 10 == 0:
                logger.info(f"迭代 {iter_num+1}/{self.max_iter}: "
                            f"最佳适应度 = {global_best_fitness:.6f}")

        optimal_centers = self._reshape_particle(global_best_particle)
        logger.info(f"\nPSO 优化完成 (最终适应度: {global_best_fitness:.6f})")
        logger.info("="*60)

        return optimal_centers, fitness_history
