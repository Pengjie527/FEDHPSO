"""
KNN缺失值插补工具
包含用于数据缺失值处理的KNN插补函数
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d


def fixgaps(x: np.ndarray) -> np.ndarray:
    """
    线性插值填补时间序列中的缺失值（NaN）
    
    在时间序列中对NaN值进行线性插值，但忽略前导和尾随的NaN值。
    参考: R. Pawlowicz 6/Nov/99
    
    Parameters:
    -----------
    x : np.ndarray
        包含NaN值的时间序列（可能为复数）
        
    Returns:
    --------
    np.ndarray
        插值后的时间序列
        
    Notes:
    ------
    - 只对中间的NaN值进行插值
    - 前导和尾随的NaN值保持不变
    """
    y = x.copy()
    bd = np.isnan(x)
    gd = np.where(~bd)[0]
    
    if len(gd) == 0:
        return y  # 全部都是NaN，无法插值
    
    bd[0:min(gd)] = 0
    bd[max(gd) + 1:] = 0
    
    if np.any(bd):
        y[bd] = interp1d(gd, x[gd], kind='linear', 
                        fill_value='extrapolate')(np.where(bd)[0])
    
    return y


def wnanmean(x: np.ndarray, weights: np.ndarray) -> float:
    """
    计算加权平均值，处理NaN值和无限权重
    
    Parameters:
    -----------
    x : np.ndarray
        数据数组
    weights : np.ndarray
        权重数组
        
    Returns:
    --------
    float
        加权平均值，如果所有值都是NaN则返回NaN
    """
    x = x.copy()
    weights = weights.copy()
    nans = np.isnan(x)
    infs = np.isinf(weights)
    
    if np.all(nans):
        return np.nan
    if np.any(infs):
        return np.nanmean(x[infs])
    
    x[nans] = 0
    weights[nans] = 0
    weights = weights / np.sum(weights)
    return np.dot(weights.T, x)


def knnimpute(data: np.ndarray, K: int = 1, useWMean: bool = True, 
              userWeights: bool = False, distance_metric: str = 'seuclidean') -> np.ndarray:
    """
    使用K最近邻方法进行缺失值插补
    
    参考: MATLAB's knnimpute.m code
    Reference: https://github.com/ogeidix/kddcup09/blob/master/utilities/knnimpute.m
    
    Parameters:
    -----------
    data : np.ndarray
        包含缺失值（NaN）的数据矩阵，形状为 (n_samples, n_features)
    K : int, default=1
        使用的最近邻数量
    useWMean : bool, default=True
        是否使用加权平均值进行插补
    userWeights : bool, default=False
        是否使用用户自定义权重（当前未实现）
    distance_metric : str, default='seuclidean'
        距离度量方法，默认使用标准化欧氏距离
        
    Returns:
    --------
    np.ndarray
        插补后的数据矩阵
        
    Notes:
    ------
    - 使用没有缺失值的行来计算最近邻距离
    - 对于每个缺失值，找到K个最近的列（特征）并插补
    - 默认使用标准化欧氏距离（seuclidean）
    """
    # 创建数据副本用于输出
    imputed = data.copy()
    
    # 识别缺失值
    nanVals = np.isnan(data)
    
    # 使用没有NaN的行来计算最近邻
    noNans = (np.sum(nanVals, axis=1) == 0)
    dataNoNans = data[noNans, :]
    
    if dataNoNans.shape[0] == 0:
        raise ValueError("所有行都包含缺失值，无法进行KNN插补")
    
    # 计算距离矩阵（转置后计算，按特征计算距离）
    distances = pdist(np.transpose(dataNoNans), distance_metric)
    
    SqF = squareform(distances)
    
    temp = SqF - np.identity(SqF.shape[0])
    
    dists = np.transpose(np.sort(temp))
    
    ndx = np.transpose(np.argsort(temp, kind='stable'))
    
    equalDists = np.vstack([
        np.diff(dists[1:, :], axis=0) == 0.0,
        np.full(dists.shape[1], False)
    ])
    
    rows = np.where(np.transpose(nanVals))[1]
    cols = np.where(np.transpose(nanVals))[0]
    
    for count in range(rows.size):
        for nearest in range(1, ndx.shape[0] - K + 1):
            L = np.where(equalDists[nearest + K - 2:, cols[count]] == 0)[0]
            if len(L) == 0:
                continue
            L = L[0]
            dataVals = data[rows[count], ndx[nearest:nearest + K + L, cols[count]]]
            if useWMean:
                if not userWeights:
                    weights = 1 / (dists[1:K + L + 1, cols[count]] + 1e-10)  # 添加小值避免除零
                val = wnanmean(dataVals, weights)
            else:
                val = np.nanmean(dataVals)
            if not np.isnan(val):
                imputed[rows[count], cols[count]] = val
                break
    
    return imputed

