import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def get_feature_definitions() -> Tuple[List[str], List[str], List[str]]:
    binary_features = ['gender', 'mechvent', 're_admission']

    # 正态分布特征 (标准化)
    normal_features = [
        'age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP',
        'RR', 'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride',
        'Glucose', 'Magnesium', 'Calcium', 'Ionised_Ca', 'CO2_mEqL',
        'Shock_Index', 'PaO2_FiO2'
    ]

    # 对数正态分布特征 (log(0.1+x)后标准化)
    lognormal_features = [
        'SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili',
        'INR', 'input_total', 'input_4hourly', 'output_total',
        'output_4hourly', 'Hb', 'WBC_count', 'Platelets_count', 'PTT',
        'PT', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3',
        'Arterial_lactate', 'Albumin', 'cumulated_balance'
    ]

    return binary_features, normal_features, lognormal_features


def load_and_filter_data(
        data_path: str,
        use_first_bloc_only: bool = True,
        logger_instance: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    加载数据并筛选第一个bloc（如果需要）
    """
    log = logger_instance if logger_instance is not None else logger
    log.info(f"正在加载数据: {data_path}")
    df_raw = pd.read_csv(data_path)

    log.info(f"  原始数据: {len(df_raw)} 条记录")

    # 如果需要只使用第一条bloc数据
    if use_first_bloc_only:
        if 'icustayid' in df_raw.columns and 'bloc' in df_raw.columns:
            # 找出每个患者的最小bloc值（第一个bloc）
            first_blocs = df_raw.groupby('icustayid')['bloc'].min().reset_index()
            first_blocs.columns = ['icustayid', 'first_bloc']

            # 合并并筛选第一个bloc的数据
            df_with_first = df_raw.merge(first_blocs, on='icustayid')
            df_filtered = df_with_first[df_with_first['bloc'] == df_with_first['first_bloc']].copy()

            # 删除辅助列
            if 'first_bloc' in df_filtered.columns:
                df_filtered = df_filtered.drop('first_bloc', axis=1)

            # 处理可能的重复（同一患者的第一个bloc有多条记录的情况）
            if df_filtered.groupby('icustayid').size().max() > 1:
                log.warning(f"  发现部分患者在第一个bloc有多条记录，保留第一条")
                df_filtered = df_filtered.groupby('icustayid').first().reset_index()

            n_patients = df_filtered['icustayid'].nunique()
            log.info(f"  提取第一个bloc数据: {len(df_filtered)} 条记录 ({n_patients} 个患者)")
            return df_filtered
        else:
            log.warning(f"  警告: 数据中缺少 'icustayid' 或 'bloc' 列，使用全部数据")
            return df_raw.copy()
    else:
        log.info(f"  使用全部数据: {len(df_raw)} 条记录")
        return df_raw.copy()


def select_features(
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        logger_instance: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    选择特征列
    """
    log = logger_instance if logger_instance is not None else logger

    if feature_columns is None:
        binary_features, normal_features, lognormal_features = get_feature_definitions()

        # 组合所有需要的特征
        important_features = binary_features + normal_features + lognormal_features

        # 从实际数据中选择存在的特征
        available_features = [col for col in important_features if col in data.columns]

        # 警告缺失特征
        missing_features = set(important_features) - set(available_features)
        if missing_features:
            log.warning(f"  缺少以下特征: {list(missing_features)}")

        feature_columns = available_features

    features = data[feature_columns].copy()
    return features, feature_columns


def handle_missing_values(
        features: pd.DataFrame,
        logger_instance: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    处理缺失值
    - lognormal 特征用中位数填充
    - 其他特征用均值填充
    """
    log = logger_instance if logger_instance is not None else logger

    _, _, lognormal_features = get_feature_definitions()

    # lognormal 特征用中位数填充，其余用均值
    for col in features.columns:
        if col in lognormal_features:
            features[col].fillna(features[col].median(), inplace=True)
        else:
            features[col].fillna(features[col].mean(), inplace=True)

    return features


def apply_log_transform(
        features: pd.DataFrame,
        logger_instance: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    对对数正态分布特征应用对数变换: log(0.1 + x)
    """
    log = logger_instance if logger_instance is not None else logger

    _, _, lognormal_features = get_feature_definitions()

    lognormal_cols = [col for col in lognormal_features if col in features.columns]
    for col in lognormal_cols:
        features[col] = np.log(0.1 + features[col].clip(lower=0))

    return features


def remove_outliers(
        features: pd.DataFrame,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        logger_instance: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    清理极端异常值：按分位数截断 + 替换 inf + 再做 log 变换
    """
    log = logger_instance if logger_instance is not None else logger

    # 清理极端异常值
    for col in features.columns:
        q1 = features[col].quantile(lower_quantile)
        q99 = features[col].quantile(upper_quantile)
        features[col] = features[col].clip(lower=q1, upper=q99)

    # 替换 inf 值
    features.replace([np.inf, -np.inf], 0, inplace=True)
    features.fillna(0, inplace=True)

    # 对 lognormal 特征做 log 变换
    features = apply_log_transform(features, logger_instance=log)
    return features


def remove_highly_correlated_features(
        features: pd.DataFrame,
        threshold: float = 0.9,
        logger_instance: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    按相关性去除高度冗余特征
    - 计算 Pearson 相关系数矩阵
    - 对于 |r| > threshold 的特征对，保留一个，删除另一个

    Returns
    -------
    剩余特征 DataFrame, 被删除特征名列表
    """
    log = logger_instance if logger_instance is not None else logger

    if features.shape[1] <= 1:
        return features, []

    corr_matrix = features.corr().abs()
    # 只看上三角
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if to_drop:
        log.info(f"  按相关性阈值 {threshold} 去除特征: {to_drop}")
        features = features.drop(columns=to_drop)
    else:
        log.info(f"  未发现 |r| > {threshold} 的高度相关特征")

    return features, to_drop


def standardize_features(
        features: pd.DataFrame,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
        logger_instance: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    标准化特征
    - 二进制特征：减去0.5
    - 正态和对数正态特征：Z-score 标准化
    """
    log = logger_instance if logger_instance is not None else logger

    binary_features, normal_features, lognormal_features = get_feature_definitions()

    if scaler is None:
        scaler = StandardScaler()

    # 标准化正态分布和对数正态分布特征
    to_scale = normal_features + lognormal_features
    to_scale = [col for col in to_scale if col in features.columns]

    if to_scale:
        if fit_scaler:
            features[to_scale] = scaler.fit_transform(features[to_scale])
        else:
            features[to_scale] = scaler.transform(features[to_scale])

    # 二进制特征偏移
    binary_cols = [col for col in binary_features if col in features.columns]
    if binary_cols:
        features[binary_cols] = features[binary_cols] - 0.5

    return features, scaler


class PreprocessingConfig:
    """数据预处理配置类"""

    def __init__(
            self,
            use_first_bloc_only: bool = True,
            feature_columns: Optional[List[str]] = None,
            lower_quantile: float = 0.01,
            upper_quantile: float = 0.99,
            apply_log_transform_first: bool = False,
            remove_correlated_features: bool = False,
            corr_threshold: float = 0.9,
    ):
        """
        初始化预处理配置

        Parameters
        ----------
        use_first_bloc_only : 是否只使用每个患者的第一条bloc数据
        feature_columns : 指定特征列，None 则自动按定义选择
        lower_quantile, upper_quantile : 异常值处理分位数
        apply_log_transform_first :（保留参数，目前未使用）
        remove_correlated_features : 是否按相关性去冗余
        corr_threshold : 相关性阈值（|r| > threshold 则删除一个）
        """
        self.use_first_bloc_only = use_first_bloc_only
        self.feature_columns = feature_columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.apply_log_transform_first = apply_log_transform_first
        self.remove_correlated_features = remove_correlated_features
        self.corr_threshold = corr_threshold


def preprocess_data(
        data_path: str,
        config: Optional[PreprocessingConfig] = None,
        site_id: Optional[int] = None,
        logger_instance: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], StandardScaler]:
    """
    完整的数据预处理流程

    Returns
    -------
    data : 原始数据框（含 SOFA, SIRS 等可用于后续分析）
    features : 预处理后的特征数据框（用于聚类）
    feature_columns : 特征列名称列表（与 features.columns 对应）
    scaler : 标准化器
    """
    if config is None:
        config = PreprocessingConfig()

    log = logger_instance if logger_instance is not None else logger

    site_prefix = f"站点 {site_id} " if site_id is not None else ""

    # 1. 加载和筛选数据
    data = load_and_filter_data(
        data_path,
        use_first_bloc_only=config.use_first_bloc_only,
        logger_instance=log
    )

    # 2. 选择特征
    features, feature_columns = select_features(
        data,
        feature_columns=config.feature_columns,
        logger_instance=log
    )

    # 3. 处理缺失值
    features = handle_missing_values(features, logger_instance=log)

    # 4. 清理异常值 + log 变换
    features = remove_outliers(
        features,
        lower_quantile=config.lower_quantile,
        upper_quantile=config.upper_quantile,
        logger_instance=log
    )

    # 4.5. 按相关性去除冗余特征
    if config.remove_correlated_features:
        features, dropped_corr = remove_highly_correlated_features(
            features,
            threshold=config.corr_threshold,
            logger_instance=log
        )
        # 更新特征列
        feature_columns = [col for col in feature_columns if col in features.columns]

    # 5. 标准化特征
    features, scaler = standardize_features(features, logger_instance=log)

    log.info(f"{site_prefix}数据预处理完成: {len(data)} 条记录, {len(feature_columns)} 个特征")
    log.info(f"{site_prefix}特征列: {feature_columns[:5]}... (共{len(feature_columns)}个)")

    return data, features, feature_columns, scaler
