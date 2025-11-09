"""
K最近邻搜索工具
包含用于查找K最近邻的函数
"""

import numpy as np


def iKNearestNeighbours(k: int, i: int, x: np.ndarray, input_mask: np.ndarray) -> np.ndarray:
    """
    找到点i在数组x中的k个最近邻，考虑输入掩码
    
    Parameters:
    -----------
    k : int
        要找到的最近邻数量
    i : int
        目标点的索引
    x : np.ndarray
        一维数组，包含所有点的坐标
    input_mask : np.ndarray
        布尔数组，标记哪些点可以作为候选最近邻
        
    Returns:
    --------
    np.ndarray
        最近邻点的索引数组（与原始utils.py行为一致，返回np.where的结果）
        
    Notes:
    ------
    这个函数用于在平滑算法（如LOESS）中找到给定点的k个最近邻，
    同时考虑哪些点可以作为候选（通过input_mask指定）
    
    返回格式与原始utils.py保持一致，以兼容现有代码
    """
    if np.count_nonzero(input_mask) <= k:
        idx = np.where(input_mask)
    else:
        d = np.abs(x - x[i])
        ds = np.sort(d[input_mask])
        dk = ds[int(k - 1)]
        close = (d <= dk)
        idx = np.where((close) & (input_mask))
    
    return idx

