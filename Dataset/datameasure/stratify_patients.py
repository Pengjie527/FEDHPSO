import pandas as pd
import numpy as np
import os
import math
from sklearn.model_selection import train_test_split

def _compute_counts_by_ratio(total, weights):
    """
    根据给定权重计算三个分组的整数人数，使用最大余数法以保证总数不变。
    """
    weight_sum = float(sum(weights))
    raw = [total * (w / weight_sum) for w in weights]
    floored = [int(math.floor(x)) for x in raw]
    remainder = [x - f for x, f in zip(raw, floored)]
    remaining = total - sum(floored)

    order = np.argsort(remainder)[::-1]
    result = floored[:]
    for idx in order[:remaining]:
        result[idx] += 1
    return result

def stratify_patients_by_severity():
    """
    将病人按照严重程度分层后随机分配到3个组，确保每组病情分布一致
    """
    print("开始按严重程度分层分组...")
    
    # 读取严重程度数据
    severity_file = '../severity_table.csv'
    if not os.path.exists(severity_file):
        print(f"错误：找不到严重程度文件 {severity_file}")
        return None, None, None
    
    severity_df = pd.read_csv(severity_file)
    # 统一并清洗ID与严重程度类型
    severity_df['icustayid'] = pd.to_numeric(severity_df['icustayid'], errors='coerce').astype('Int64')
    severity_df['severity'] = pd.to_numeric(severity_df['severity'], errors='coerce').astype('Int64')
    severity_df = severity_df.dropna(subset=['icustayid', 'severity'])
    severity_df['icustayid'] = severity_df['icustayid'].astype(int)
    severity_df['severity'] = severity_df['severity'].astype(int)
    print(f"严重程度数据形状: {severity_df.shape}")
    print(f"严重程度分布（原始）:")
    print(severity_df['severity'].value_counts().sort_index())
    
    # 读取MIMIC数据
    mimic_file = '../MIMICtable.csv'
    if not os.path.exists(mimic_file):
        print(f"错误：找不到MIMIC数据文件 {mimic_file}")
        return None, None, None
    
    try:
        mimic_df = pd.read_csv(mimic_file)
    except Exception as e:
        print(f"读取MIMIC数据文件时出错: {e}")
        return None, None, None
    
    # 统一ID类型
    if 'icustayid' not in mimic_df.columns:
        print("错误：MIMIC数据缺少 icustayid 列")
        return None, None, None
    mimic_df['icustayid'] = pd.to_numeric(mimic_df['icustayid'], errors='coerce').astype('Int64')
    mimic_df = mimic_df.dropna(subset=['icustayid'])
    mimic_df['icustayid'] = mimic_df['icustayid'].astype(int)
    print(f"MIMIC数据形状: {mimic_df.shape}")

    # 统计唯一病人数与交集
    n_sev_unique = severity_df['icustayid'].nunique()
    n_mimic_unique = mimic_df['icustayid'].nunique()
    intersect_ids = np.intersect1d(severity_df['icustayid'].unique(), mimic_df['icustayid'].unique())
    print(f"严重程度表唯一病人: {n_sev_unique}")
    print(f"MIMIC表唯一病人: {n_mimic_unique}")
    print(f"两表交集唯一病人: {len(intersect_ids)}")

    # 仅保留交集病人，避免无效ID影响分布
    severity_df = severity_df[severity_df['icustayid'].isin(intersect_ids)].copy()
    print(f"交集后的严重程度分布:")
    print(severity_df['severity'].value_counts().sort_index())
    
    # 按严重程度分层
    groups = []
    for severity in [1, 2, 3]:  # 1=轻度, 2=中度, 3=重度
        # 去重后分组
        severity_patients = (
            severity_df[severity_df['severity'] == severity]['icustayid']
            .drop_duplicates()
            .tolist()
        )
        print(f"\n严重程度 {severity} 的病人数量: {len(severity_patients)}")
        
        if len(severity_patients) > 0:
            # 将每个严重程度的病人按 5.5:3.5:1 比例分到3组
            np.random.seed(42)  # 设置随机种子确保结果可重复
            severity_patients = np.array(severity_patients)
            np.random.shuffle(severity_patients)  # 随机打乱

            weights = [5.5, 3.5, 1.0]
            group_sizes = _compute_counts_by_ratio(len(severity_patients), weights)

            # 分配病人到3组（按计算出的大小切片）
            start_idx = 0
            for i in range(3):
                group_size = group_sizes[i]
                end_idx = start_idx + group_size

                group_patients = severity_patients[start_idx:end_idx].tolist()
                groups.append({
                    'severity': severity,
                    'group': i + 1,
                    'patients': group_patients
                })

                print(f"  组{i+1}: {len(group_patients)} 个病人")
                start_idx = end_idx
    
    # 合并每个组的病人
    group1_patients = []
    group2_patients = []
    group3_patients = []
    
    for item in groups:
        if item['group'] == 1:
            group1_patients.extend(item['patients'])
        elif item['group'] == 2:
            group2_patients.extend(item['patients'])
        else:
            group3_patients.extend(item['patients'])
    
    # 每组总人数与严重程度分布汇总
    print(f"\n每组人数与严重程度分布汇总:")
    summary_rows = []
    for i, group_patients in enumerate([group1_patients, group2_patients, group3_patients], 1):
        group_df = severity_df[severity_df['icustayid'].isin(group_patients)]
        counts = group_df['severity'].value_counts().sort_index()
        mild = int(counts.get(1, 0))
        moderate = int(counts.get(2, 0))
        severe = int(counts.get(3, 0))
        total = len(group_patients)
        summary_rows.append({'组': f'组{i}', '总人数': total, '轻度': mild, '中度': moderate, '重度': severe})
    summary_df = pd.DataFrame(summary_rows, columns=['组', '总人数', '轻度', '中度', '重度'])
    print(summary_df.to_string(index=False))
    
    # 创建输出目录
    output_dir = '../datawithnum'
    os.makedirs(output_dir, exist_ok=True)
    # 保存汇总表
    try:
        summary_df.to_csv(f'{output_dir}/group_summary.csv', index=False, encoding='utf-8-sig')
    except Exception:
        summary_df.to_csv(f'{output_dir}/group_summary.csv', index=False)
    
    # 分割MIMIC数据
    group1_data = mimic_df[mimic_df['icustayid'].isin(group1_patients)]
    group2_data = mimic_df[mimic_df['icustayid'].isin(group2_patients)]
    group3_data = mimic_df[mimic_df['icustayid'].isin(group3_patients)]
    
    print(f"\n数据分割结果:")
    print(f"组1数据形状: {group1_data.shape}")
    print(f"组2数据形状: {group2_data.shape}")
    print(f"组3数据形状: {group3_data.shape}")
    
    # 保存数据
    group1_data.to_csv(f'{output_dir}/group1_mimic_data.csv', index=False)
    group2_data.to_csv(f'{output_dir}/group2_mimic_data.csv', index=False)
    group3_data.to_csv(f'{output_dir}/group3_mimic_data.csv', index=False)
    
    # 保存病人ID列表
    pd.DataFrame({'icustayid': group1_patients}).to_csv(f'{output_dir}/group1_patients.csv', index=False)
    pd.DataFrame({'icustayid': group2_patients}).to_csv(f'{output_dir}/group2_patients.csv', index=False)
    pd.DataFrame({'icustayid': group3_patients}).to_csv(f'{output_dir}/group3_patients.csv', index=False)
    
    print(f"\n数据分割完成！文件已保存到 '{output_dir}' 目录中。")
    
    return group1_data, group2_data, group3_data

if __name__ == "__main__":
    stratify_patients_by_severity()
