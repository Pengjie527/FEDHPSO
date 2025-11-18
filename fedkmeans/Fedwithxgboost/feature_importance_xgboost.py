import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Any
from sklearn.model_selection import train_test_split

# 导入特征处理模块
from featuredatameasure.preprocessing import preprocess_data, PreprocessingConfig

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sepsis_label(data: pd.DataFrame) -> pd.DataFrame:
    """
    生成sepsis_label特征
    
    基于以下规则生成sepsis标签：
    - 如果患者有sepsis相关诊断或治疗指标异常，则标记为1
    
    Args:
        data: 包含患者信息的DataFrame
    
    Returns:
        添加了sepsis_label列的DataFrame
    """
    # 定义可能指示sepsis的列（这些列名需要根据实际数据集进行调整）
    sepsis_indicators = [
        'sepsis', 'septicemia', 'sepsis_dx', 'blood_culture_positive',
        'sirs_score', 'qsofa_score', 'sirs_criteria', 'lactate',
        'wbc_count', 'temperature', 'heart_rate', 'respiratory_rate',
        'systolic_bp', 'mean_arterial_pressure'
    ]
    
    # 初始化sepsis_label列为0
    data['sepsis_label'] = 0
    
    # 检查数据中存在的指标列
    available_indicators = [col for col in sepsis_indicators if col in data.columns]
    logger.info(f"检测到 {len(available_indicators)} 个可用的sepsis指示列: {available_indicators}")
    
    # 基于不同的指标设置sepsis标签
    for col in available_indicators:
        try:
            # 处理直接的sepsis标记列（布尔或0/1值）
            if col in ['sepsis', 'septicemia', 'sepsis_dx', 'blood_culture_positive']:
                if data[col].dtype == 'object':
                    # 如果是字符串类型，检查是否包含阳性词汇
                    data.loc[data[col].str.contains('yes|positive|true|1', case=False, na=False), 'sepsis_label'] = 1
                else:
                    # 如果是数值类型，检查是否为1或True
                    data.loc[data[col] == 1, 'sepsis_label'] = 1
                    data.loc[data[col] == True, 'sepsis_label'] = 1
            
            # 处理评分列
            elif col in ['sirs_score', 'qsofa_score']:
                # SIRS评分≥2或qSOFA评分≥2通常指示感染严重程度
                if col == 'sirs_score':
                    data.loc[data[col] >= 2, 'sepsis_label'] = 1
                else:  # qsofa_score
                    data.loc[data[col] >= 2, 'sepsis_label'] = 1
            
            # 处理生理指标
            elif col == 'lactate':
                # 乳酸≥2 mmol/L可能指示组织灌注不足
                data.loc[data[col] >= 2.0, 'sepsis_label'] = 1
            
            elif col == 'wbc_count':
                # 白细胞计数异常（>12000/μL或<4000/μL）
                data.loc[(data[col] > 12000) | (data[col] < 4000), 'sepsis_label'] = 1
            
            elif col == 'temperature':
                # 体温异常（>38°C或<36°C）
                data.loc[(data[col] > 38.0) | (data[col] < 36.0), 'sepsis_label'] = 1
            
            elif col == 'heart_rate':
                # 心率>90次/分
                data.loc[data[col] > 90, 'sepsis_label'] = 1
            
            elif col == 'respiratory_rate':
                # 呼吸频率>20次/分
                data.loc[data[col] > 20, 'sepsis_label'] = 1
            
            elif col in ['systolic_bp', 'mean_arterial_pressure']:
                # 低血压（收缩压<90mmHg或平均动脉压<65mmHg）
                if col == 'systolic_bp':
                    data.loc[data[col] < 90, 'sepsis_label'] = 1
                else:  # mean_arterial_pressure
                    data.loc[data[col] < 65, 'sepsis_label'] = 1
        
        except Exception as e:
            logger.warning(f"处理sepsis指标列 '{col}' 时出错: {str(e)}")
    
    # 统计结果
    sepsis_count = data['sepsis_label'].sum()
    logger.info(f"生成sepsis_label特征: 阳性样本 {sepsis_count} ({sepsis_count/len(data)*100:.2f}%)")
    
    return data

def setup_plot_style():
    """设置绘图样式"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-darkgrid')



def fitness_function(X, y, selected_features: np.ndarray, opts: Dict = None) -> float:
    """
    适应度函数：基于KNN的多目标评估（根据Matlab文件tNewFitnessFunction.m的实现）
    
    Args:
        X: 特征矩阵（numpy数组或pandas DataFrame）
        y: 标签向量
        selected_features: 特征选择掩码（布尔数组）
        opts: 可选参数
    
    Returns:
        适应度值（越小越好）：加权组合的多目标适应度值
    """
    # 默认权重参数（对应Matlab中的ws）
    ws = np.array([0.98, 0.01, 0.01])  # [alpha, beta, gamma]
    
    # 从opts获取参数
    if opts is not None:
        if 'ws' in opts:
            ws = np.array(opts['ws'])
        
    # 检查是否有特征被选中
    if not np.any(selected_features):
        logger.warning("没有选择任何特征")
        return 1.0  # 无特征时返回最差适应度
    
    # 确保X是numpy数组
    if hasattr(X, 'values'):  # 如果是pandas DataFrame
        X_array = X.values
    elif isinstance(X, pd.Series):
        X_array = X.values.reshape(-1, 1)
    else:
        X_array = np.array(X)
    
    # 确保X_array是二维数组
    if X_array.ndim == 1:
        X_array = X_array.reshape(-1, 1)
    
    # 确保selected_features维度匹配
    if len(selected_features) != X_array.shape[1]:
        logger.error(f"特征掩码维度不匹配: {len(selected_features)} vs {X_array.shape[1]}")
        return 1.0  # 维度不匹配时返回最差适应度
    
    # 使用选择的特征
    X_selected = X_array[:, selected_features]
    
    # 确保有足够的特征
    if X_selected.shape[1] == 0:
        return 1.0  # 无特征时返回最差适应度
    
    # 确保y是numpy数组
    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    try:
        # 获取K值，默认5
        k = 5
        if opts is not None and 'k' in opts:
            k = opts['k']
        
        # 处理数据划分
        if opts is not None and 'Model' in opts:
            # 如果提供了预定义的训练/测试集（对应Matlab中的Model结构）
            Model = opts['Model']
            if hasattr(Model, 'training') and hasattr(Model, 'test'):
                trainIdx = Model.training
                testIdx = Model.test
                X_train = X_selected[trainIdx]
                y_train = y_array[trainIdx]
                X_val = X_selected[testIdx]
                y_val = y_array[testIdx]
            else:
                # 默认划分训练集和验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    X_selected, y_array, test_size=0.2, random_state=42, 
                    stratify=y_array if len(np.unique(y_array)) > 1 else None
                )
        else:
            # 默认划分训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_selected, y_array, test_size=0.2, random_state=42, 
                stratify=y_array if len(np.unique(y_array)) > 1 else None
            )
        
        # 训练KNN模型
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import f1_score, confusion_matrix
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # 预测
        y_pred = knn.predict(X_val)
        y_pred_prob = knn.predict_proba(X_val)
        
        # 计算错误率（对应Matlab中的error）
        # 1. 计算F1分数作为评估指标
        try:
            # 获取所有唯一类别
            unique_classes = np.unique(y_val)
            if len(unique_classes) > 1:
                # 多类别情况下使用macro F1
                f1 = f1_score(y_val, y_pred, average='macro')
            else:
                # 二分类情况下使用标准F1
                f1 = f1_score(y_val, y_pred)
            error = 1.0 - f1  # 转换为错误率
        except Exception:
            # 如果F1计算失败，使用准确率作为备选
            accuracy = np.mean(y_pred == y_val)
            error = 1.0 - accuracy
        
        # 计算特征数量比例
        num_feat = np.sum(selected_features)
        max_feat = len(selected_features)
        feat_ratio = num_feat / max_feat
        
        # 计算边界样本比例（对应Matlab中的min_border）
        # 使用KNN查找每个测试样本的邻居，并计算反向邻居的数量
        min_border = 0.0
        try:
            # 使用sklearn的KNN来查找邻居
            from sklearn.neighbors import NearestNeighbors
            
            # 为测试集构建最近邻模型
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_val)
            distances, indices = nbrs.kneighbors(X_val)
            
            # 计算每个样本的反向邻居数量
            num_rev_neighbour = np.zeros(len(y_val))
            for i in range(len(y_val)):
                # 检查k个邻居（除了自己）
                for j in range(1, k):
                    if y_val[indices[i, j]] != y_val[i]:
                        num_rev_neighbour[i] += 1
            
            # 找出边界样本（反向邻居数超过(k-1)*0.5的样本）
            border_samples = np.sum(num_rev_neighbour > (k-1)*0.5)
            min_border = border_samples / len(y_val)
        except Exception as e:
            logger.warning(f"边界样本计算失败: {str(e)}")
            min_border = 0.0  # 计算失败时使用默认值
        
        # 计算加权适应度值
        alpha = ws[0]
        beta = ws[1]
        gamma = ws[2]
        fitness = alpha * error + beta * feat_ratio + gamma * min_border
        
        logger.debug(f"适应度计算 - 错误率: {error:.4f}, 特征比例: {feat_ratio:.4f}, 边界比例: {min_border:.4f}, 适应度: {fitness:.4f}")
        
        return fitness
    except Exception as e:
        logger.error(f"适应度函数计算错误: {str(e)}")
        return 1.0  # 异常情况下返回最差适应度

def thres_with_penalty(arc_wf: np.ndarray, arc_bf: np.ndarray, len_div: int, 
                       thres_ub: float, thres_lb: float, ii: int) -> np.ndarray:
    """
    计算带有惩罚的阈值
    
    Args:
        arc_wf: 特征被选为差特征的计数
        arc_bf: 特征被选为好特征的计数
        len_div: 特征维度
        thres_ub: 阈值上界
        thres_lb: 阈值下界
        ii: 当前分区索引
    
    Returns:
        计算得到的阈值数组
    """
    # 初始化阈值
    thres = np.zeros(len_div)
    for d in range(len_div):
        # 根据分区索引动态调整基础阈值
        thres[d] = (thres_ub - thres_lb) * (2 * np.arctan(ii) / np.pi) + thres_lb
    
    # 根据历史选择情况调整阈值
    for d in range(len_div):
        # 计算差特征的百分比
        total_count = arc_wf[d] + arc_bf[d] + 1e-10  # 避免除零
        per_of_wf = arc_wf[d] / total_count
        
        # 根据历史表现调整阈值
        if per_of_wf > 0.7:
            thres[d] = 0.7  # 难以选择的特征，提高阈值
        elif per_of_wf > 0.3:
            thres[d] = 0.3 # 表现一般的特征，稍微提高阈值
    
    return thres

def find_better_X(X: np.ndarray, X_new: np.ndarray, X_data: np.ndarray, y_data: np.ndarray, 
                 thres: np.ndarray, fitness_func, opts: Dict = None) -> np.ndarray:
    """
    贪婪策略选择更好的位置
    
    Args:
        X: 当前位置
        X_new: 新位置
        X_data: 特征数据
        y_data: 标签数据
        thres: 阈值数组
        fitness_func: 适应度函数
        opts: 可选参数
    
    Returns:
        更好的位置
    """
    # 计算当前位置的适应度
    current_fitness = fitness_func(X_data, y_data, X > thres, opts)
    # 计算新位置的适应度
    new_fitness = fitness_func(X_data, y_data, X_new > thres, opts)
    
    # 如果新位置更好，返回新位置
    if new_fitness < current_fitness:
        return X_new
    return X

def bacterial_foraging_feature_selection(X: np.ndarray, y: np.ndarray, opts: Dict = None) -> Tuple[np.ndarray, List[float]]:
    """
    基于细菌觅食优化的双阈值启发式特征选择
    
    Args:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,)
        opts: 算法参数
    
    Returns:
        选中的特征索引和适应度曲线
    """
    # 默认参数
    if opts is None:
        opts = {}
    
    # 基本参数
    lb = opts.get('lb', 0.0)
    ub = opts.get('ub', 1.0)
    c = opts.get('c', 0.5)  # 趋化步长
    Nc = opts.get('Nc', 20)  # 趋化次数
    Nre = opts.get('Nre', 5)  # 繁殖次数
    Fre = opts.get('Fre', Nc)  # 繁殖频率
    Fed = opts.get('Fed', Nre * Fre)  # 消除-扩散频率
    Ped = opts.get('Ped', 0.25)  # 消除-扩散概率
    N = opts.get('N', 30)  # 细菌总数
    max_iter = opts.get('T', 30)  # 最大迭代次数
    NbrDiv = opts.get('NbrDiv', 3)  # 分区数量
    
    # 确保X是二维数组
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # 数据维度
    n_samples, dim = X.shape
    logger.info(f"输入数据维度: {X.shape}, 特征数量: {dim}")
    
    # 调整分区数量以适应特征数量
    if dim < NbrDiv:
        NbrDiv = max(1, dim)
    
    # 分区参数
    DivSize = N // NbrDiv  # 每个分区的细菌数量
    if DivSize < 2:
        DivSize = 2
        N = NbrDiv * DivSize
    Sr = DivSize // 2  # 繁殖时保留的细菌数量
    bin_Div = dim // NbrDiv  # 每个分区的基础维度
    
    # 计算每个分区的实际维度
    len_Div = np.zeros(NbrDiv, dtype=int)
    for ii in range(NbrDiv - 1):
        len_Div[ii] = bin_Div
    len_Div[-1] = dim - (NbrDiv - 1) * bin_Div
    
    logger.info(f"分区数量: {NbrDiv}, 每个分区维度: {len_Div.tolist()}")
    
    # 阈值参数
    thres_ub = opts.get('thres_ub', 1.0)
    thres_lb = opts.get('thres_lb', 0.5)
    
    # 历史记录：好特征和差特征的计数
    arc_wf = np.zeros(dim)  # 差特征计数
    arc_bf = np.zeros(dim)  # 好特征计数
    
    # 初始化种群
    pop = {}
    pop['position'] = []  # 每个分区的细菌位置
    pop['fit'] = np.full((NbrDiv, DivSize), np.inf)  # 适应度值
    pop['Div_AF'] = np.zeros(NbrDiv)  # 每个分区的平均适应度
    
    for ii in range(NbrDiv):
        # 初始化当前分区的细菌位置
        positions = np.random.uniform(lb, ub, (DivSize, len_Div[ii]))
        pop['position'].append(positions)
    
    # 全局最优和各分区最优初始化
    fitG = np.inf
    fitG_div = np.full(NbrDiv, np.inf)
    Xb = [None] * NbrDiv
    best_thres = [0.5] * NbrDiv  # 记录每个分区的最佳阈值
    
    # 首次评估适应度
    for ii in range(NbrDiv):
        for i in range(DivSize):
            current_pos = pop['position'][ii][i, :]
            
            # 计算阈值 - 只使用当前分区的历史记录
            start_idx = sum(len_Div[:ii])
            end_idx = start_idx + len_Div[ii]
            
            # 提取当前分区的历史记录
            arc_wf_div = arc_wf[start_idx:end_idx]
            arc_bf_div = arc_bf[start_idx:end_idx]
            
            thres = thres_with_penalty(arc_wf_div, arc_bf_div, len_Div[ii], thres_ub, thres_lb, ii + 1)
            
            # 计算适应度
            # 构造完整的特征选择掩码
            full_mask = np.zeros(dim, dtype=bool)
            full_mask[start_idx:end_idx] = current_pos > thres
            
            # 使用原始X数据计算适应度
            fit = fitness_function(X, y, full_mask, opts)
            pop['fit'][ii, i] = fit
            
            # 更新全局最优
            if fit < fitG:
                fitG = fit
                logger.info(f"新的全局最优适应度: {fitG:.4f}")
            
            # 更新分区最优
            if fit < fitG_div[ii]:
                Xb[ii] = current_pos.copy()
                fitG_div[ii] = fit
        
        # 更新分区平均适应度
        pop['Div_AF'][ii] = np.mean(pop['fit'][ii])
    
    # 记录适应度曲线
    curve = [fitG]
    
    # 主迭代循环
    for t in range(1, max_iter + 1):
        logger.info(f"迭代 {t}/{max_iter}, 当前最佳适应度: {fitG:.4f}")
        
        for ii in range(NbrDiv):
            # 获取当前分区的细菌
            positions = pop['position'][ii]
            start_idx = sum(len_Div[:ii])
            end_idx = start_idx + len_Div[ii]
            
            for i in range(DivSize):
                # 趋化过程
                # 生成随机方向
                Delta = (2 * np.random.randint(0, 2, size=len_Div[ii]) - 1) * np.random.rand(len_Div[ii])
                norm_Delta = np.linalg.norm(Delta) + 1e-10  # 避免除零
                PHI = Delta / norm_Delta
                
                # 更新位置
                new_pos = positions[i] + c * PHI
                # 边界处理
                new_pos = np.clip(new_pos, lb, ub)
                positions[i] = new_pos
                
                # 计算阈值 - 只使用当前分区的历史记录
                arc_wf_div = arc_wf[start_idx:end_idx]
                arc_bf_div = arc_bf[start_idx:end_idx]
                thres = thres_with_penalty(arc_wf_div, arc_bf_div, len_Div[ii], thres_ub, thres_lb, ii + 1)
                
                # 计算适应度
                full_mask = np.zeros(dim, dtype=bool)
                full_mask[start_idx:end_idx] = new_pos > thres
                
                fit = fitness_function(X, y, full_mask, opts)
                pop['fit'][ii, i] = fit
                
                # 记录特征选择历史
                arc_tf = np.where(full_mask)[0]  # 当前选中的特征索引
                if fit < pop['Div_AF'][ii]:
                    arc_bf[arc_tf] += 1  # 好特征计数增加
                else:
                    arc_wf[arc_tf] += 1  # 差特征计数增加
                
                # 更新全局最优
                if fit < fitG:
                    fitG = fit
                    logger.debug(f"迭代 {t} - 新的全局最优适应度: {fitG:.4f}")
                
                # 更新分区最优
                if fit < fitG_div[ii]:
                    Xb[ii] = new_pos.copy()
                    fitG_div[ii] = fit
                    best_thres[ii] = thres  # 记录最佳阈值
            
            # 更新分区平均适应度
            pop['Div_AF'][ii] = np.mean(pop['fit'][ii])
            
            # 繁殖过程
            if t % Fre == 0:
                # 按适应度排序
                sorted_indices = np.argsort(pop['fit'][ii])
                positions = positions[sorted_indices]
                
                # 保留前Sr个，复制到后Sr个位置
                for jj in range(Sr):
                    positions[jj + Sr] = positions[jj].copy()
                
                pop['position'][ii] = positions
            
            # 消除-扩散过程
            if t % Fed == 0:
                for m in range(DivSize):
                    if np.random.rand() < Ped:
                        # 生成新位置
                        X_new = np.random.uniform(lb, ub, size=len_Div[ii])
                        
                        # 使用贪婪策略选择
                        positions[m] = find_better_X(positions[m], X_new, 
                                                   X, y, thres, fitness_function, opts)
                
                pop['position'][ii] = positions
        
        # 记录当前迭代的最佳适应度
        curve.append(fitG)
    
    # 生成最终的特征选择掩码（基于BFO算法的全局最佳位置和动态阈值）
    feature_mask = np.zeros(dim, dtype=bool)
    selected_indices = np.array([])
    
    for i in range(NbrDiv):
        if Xb[i] is not None:
            # 计算该分区的起始和结束索引
            dim_per_div = dim // NbrDiv
            remaining_dim = dim % NbrDiv
            start_idx = i * dim_per_div + min(i, remaining_dim)
            end_idx = start_idx + len_Div[i]  # 使用预先计算的分区长度
            
            # 直接生成分区的特征掩码
            partition_mask = Xb[i] > best_thres[i]
            feature_mask[start_idx:end_idx] = partition_mask
            
            # 收集选中的索引（作为辅助信息）
            local_selected = np.where(partition_mask)[0] + start_idx
            # 计算当前分区的平均阈值用于日志记录
            avg_thres = np.mean(best_thres[i]) if isinstance(best_thres[i], np.ndarray) else best_thres[i]
            logger.debug(f"分区 {i+1} - 平均阈值: {avg_thres:.4f}, 选中特征数: {len(local_selected)}")
            selected_indices = np.union1d(selected_indices, local_selected)
    
    selected_indices = selected_indices.astype(int)
    selected_count = np.sum(feature_mask)
    logger.info(f"BFO算法选择的特征数量: {selected_count}")
    
    # 直接返回特征掩码（主要结果）和索引（辅助信息）以及适应度曲线
    return feature_mask, selected_indices, curve



def plot_feature_importance(top_features: List[Tuple[str, float]], output_dir: str = './results'):
    """
    绘制特征重要性图
    
    Args:
        top_features: 包含(特征名, 重要性得分)的列表
        output_dir: 结果保存目录
    """
    setup_plot_style()
    
    features, importance = zip(*top_features)
    plt.figure(figsize=(12, max(6, len(top_features) * 0.3)))
    
    # 水平条形图
    bars = plt.barh(features, importance, color='skyblue')
    plt.xlabel('重要性得分')
    plt.title('梯度提升特征重要性 (Top Features)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', va='center')
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"特征重要性图已保存至: {save_path}")
    
    plt.close()

def main(data_path: str = '/Users/bytedance/fedhpso/FEDHPSO/Dataset/MIMICtable_subset_5000.csv',
         target_col: str = 'died_in_hosp',
         output_dir: str = './results_fedh_pso',
         save_selected_features_file: bool = True,
         generate_sepsis: bool = True):
    """
    主函数：使用双阈值启发式方法选择特征
    
    Args:
        data_path: 数据文件路径
        target_col: 目标变量列名
        output_dir: 结果保存目录
        save_selected_features_file: 是否保存选中的特征名称到文件
        generate_sepsis: 是否生成sepsis_label特征
    """
    logger.info(f"开始使用双阈值启发式方法进行特征选择分析，目标变量: {target_col}")
    
    # 配置预处理
    preprocess_config = PreprocessingConfig(
        use_first_bloc_only=True,  # 只使用每个患者的第一条记录
        remove_correlated_features=True,
        corr_threshold=0.9
    )
    
    # 预处理数据
    logger.info(f"预处理数据: {data_path}")
    data, features, feature_columns, scaler = preprocess_data(
        data_path=data_path,
        config=preprocess_config,
        logger_instance=logger
    )
    
    # 生成sepsis_label特征（如果需要）
    if generate_sepsis:
        logger.info("生成sepsis_label特征...")
        data = generate_sepsis_label(data)
        
        # 将sepsis_label添加到特征矩阵中
        # 首先检查是否已经存在sepsis_label在特征列中
        if 'sepsis_label' not in feature_columns:
            # 获取sepsis_label的特征值
            sepsis_feature = data['sepsis_label'].values.reshape(-1, 1)
            
            # 将sepsis_label添加到特征矩阵
            features = np.hstack([features, sepsis_feature])
            
            # 更新特征列名列表
            feature_columns = list(feature_columns) + ['sepsis_label']
            
            logger.info("已将'sepsis_label'添加到特征矩阵中")
        else:
            logger.info("'sepsis_label'特征已存在于特征列中")
    
    # 检查目标变量是否存在
    if target_col not in data.columns:
        raise ValueError(f"目标变量 '{target_col}' 在数据中不存在")
    
    # 获取目标变量
    y = data[target_col]
    logger.info(f"目标变量 '{target_col}' 分布: {y.value_counts().to_dict()}")
    
    # 设置BFO算法参数
    bfo_opts = {
        'N': 10,  # 细菌总数
        'NbrDiv': 1,  # 分区数量
        'T': 20,  # 最大迭代次数
        'c': 0.6,  # 趋化步长
        'Ped': 0.2,  # 消除-扩散概率
        'thres_ub': 0.9,  # 阈值上界
        'thres_lb': 0.4  # 阈值下界
    }
    
    # 使用双阈值启发式方法选择特征
    logger.info("开始使用双阈值启发式方法进行特征选择...")
    feature_mask, selected_indices, curve = bacterial_foraging_feature_selection(features, y, bfo_opts)
    
    # 创建top_features列表用于后续展示和保存
    # 对于选择的特征，我们赋予相同的重要性得分
    top_features = []
    selected_count = np.sum(feature_mask)
    
    if selected_count > 0:
        importance_score = 1.0 / selected_count
        # 直接使用特征掩码选择特征，避免索引不匹配问题
        top_features = [(feature_columns[i], importance_score) for i in range(len(feature_columns)) if feature_mask[i]]
    else:
        logger.warning("BFO算法未选择任何特征")
    
    # 获取选中的特征名称列表
    selected_feature_names = [feature_columns[i] for i in range(len(feature_columns)) if feature_mask[i]]
    

    
    logger.info(f"选中的特征数量: {len(top_features)}")
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"  {i:2d}. {feature}: {importance:.6f}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存BFO特征选择结果
    results = {
        'target_variable': target_col,
        'selected_features': [{'feature': f, 'importance': float(i)} for f, i in top_features],
        'bfo_parameters': bfo_opts,
        'fitness_curve': curve,
        'selected_feature_count': len(selected_indices)
    }
    
    results_path = os.path.join(output_dir, 'bfo_feature_selection.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"BFO结果已保存至: {results_path}")
    
    # 保存选中的特征名称到文件
    if save_selected_features_file and selected_feature_names:
        features_path = os.path.join(output_dir, 'selected_features.txt')
        save_selected_features(selected_feature_names, features_path)
    
    # 绘制特征重要性图
    plot_feature_importance(top_features, output_dir)
    
    # 返回结果
    return results

def get_feature_names_from_data(data_path: str, config: PreprocessingConfig = None, logger_instance = None) -> List[str]:
    """
    从数据文件中获取特征名称列表
    
    Args:
        data_path: 数据文件路径
        config: 预处理配置
        logger_instance: 日志实例
    
    Returns:
        特征名称列表
    """
    if logger_instance is None:
        logger_instance = logging.getLogger(__name__)
    
    if config is None:
        config = PreprocessingConfig(
            use_first_bloc_only=True,
            remove_correlated_features=True,
            corr_threshold=0.9
        )
    
    logger_instance.info(f"获取特征名称: {data_path}")
    # 预处理数据，但只返回特征名称
    _, _, feature_columns, _ = preprocess_data(
        data_path=data_path,
        config=config,
        logger_instance=logger_instance
    )
    
    return feature_columns

def save_selected_features(selected_features: List[str], output_path: str):
    """
    保存选中的特征名称到文件
    
    Args:
        selected_features: 选中的特征名称列表
        output_path: 输出文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存特征名称
    with open(output_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    logger.info(f"已保存 {len(selected_features)} 个选中的特征到: {output_path}")

# 修改main函数以支持保存选中特征
if __name__ == '__main__':
    # 运行主函数
    main(
        data_path='/Users/bytedance/fedhpso/FEDHPSO/Dataset/datawithnum/group1_mimic_data.csv',
        target_col='died_in_hosp'
        # 不再需要top_k参数，直接使用算法选择的所有特征
    )