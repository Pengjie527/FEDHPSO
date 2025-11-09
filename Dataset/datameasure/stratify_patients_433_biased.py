import pandas as pd
import numpy as np
import os
import math


def _compute_counts_by_ratio(total, weights):
    """
    根据给定权重计算三个分组的整数人数，使用最大余数法以保证总数不变。
    """
    weight_sum = float(sum(weights))
    if weight_sum == 0:
        return [0, 0, 0]
    raw = [total * (w / weight_sum) for w in weights]
    floored = [int(math.floor(x)) for x in raw]
    remainder = [x - f for x, f in zip(raw, floored)]
    remaining = total - sum(floored)

    order = np.argsort(remainder)[::-1]
    result = floored[:]
    for idx in order[:remaining]:
        result[idx] += 1
    return result


def _allocate_with_bias(total_count, weights_pref, remaining_caps):
    """
    在给定的剩余容量 remaining_caps 下，按偏好权重 weights_pref 分配 total_count 个元素。
    先用最大余数法得到建议分配，再按剩余容量裁剪并将余量按偏好顺序回填。
    返回长度为3的分配列表。
    """
    if total_count <= 0:
        return [0, 0, 0]

    # 初步分配
    proposed = _compute_counts_by_ratio(total_count, weights_pref)

    # 按容量裁剪
    clipped = [min(p, remaining_caps[i]) for i, p in enumerate(proposed)]
    assigned = sum(clipped)
    leftover = total_count - assigned

    # 余量按权重偏好回填（容量优先）
    if leftover > 0:
        # 偏好顺序：权重从大到小；容量更大的更优先
        order = sorted(range(3), key=lambda i: (weights_pref[i], remaining_caps[i] - clipped[i]), reverse=True)
        idx_pointer = 0
        while leftover > 0 and any(remaining_caps[i] - clipped[i] > 0 for i in range(3)):
            i = order[idx_pointer % 3]
            if remaining_caps[i] - clipped[i] > 0:
                clipped[i] += 1
                leftover -= 1
            idx_pointer += 1

    return clipped


def stratify_patients_by_severity_biased_433():
    """
    将病人按严重程度分层，并分到3个组，总人数比例约 4:3:3。
    - 组1（4份）：重症（severity=3）相对更多
    - 组3（最后一个3份）：轻症（severity=1）相对更多
    - 组2：中性填充
    """
    print("开始按严重程度分层分组（4:3:3 且带偏好）...")

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

    # 仅保留交集病人
    severity_df = severity_df[severity_df['icustayid'].isin(intersect_ids)].copy()
    print(f"交集后的严重程度分布:")
    print(severity_df['severity'].value_counts().sort_index())

    # 目标总人数比例 4:3:3
    all_unique_ids = np.unique(severity_df['icustayid'].values)
    total_n = len(all_unique_ids)
    target_group_sizes = _compute_counts_by_ratio(total_n, [4, 3, 3])
    group_remaining = target_group_sizes[:]
    print(f"目标组容量(4:3:3): {target_group_sizes}")

    # 分层后分配（按偏好权重且受总容量约束）
    np.random.seed(42)
    groups = []

    # 严重程度优先顺序：先重症，再轻症，最后中度（中度用于平衡容量）
    severity_order = [3, 1, 2]
    severity_to_weights = {
        3: [6, 3, 1],  # 重症优先进组1
        1: [1, 3, 6],  # 轻症优先进组3
        2: [4, 3, 3],  # 中度按总体容量趋向填充
    }

    for severity in severity_order:
        # 去重并获取该层病人
        severity_patients = (
            severity_df[severity_df['severity'] == severity]['icustayid']
            .drop_duplicates()
            .tolist()
        )
        print(f"\n严重程度 {severity} 的病人数量: {len(severity_patients)}")

        if len(severity_patients) == 0:
            continue

        patients_arr = np.array(severity_patients)
        np.random.shuffle(patients_arr)

        # 计算该严重度在三个组的分配数量（受剩余容量约束）
        weights_pref = severity_to_weights[severity]
        alloc_counts = _allocate_with_bias(len(patients_arr), weights_pref, group_remaining)

        # 切片分配
        start_idx = 0
        for i in range(3):
            count_i = alloc_counts[i]
            end_idx = start_idx + count_i
            assigned_ids = patients_arr[start_idx:end_idx].tolist()
            groups.append({
                'severity': severity,
                'group': i + 1,
                'patients': assigned_ids,
            })
            group_remaining[i] -= count_i
            start_idx = end_idx
            print(f"  组{i+1}: 分到 {count_i} 个病人（剩余容量 {group_remaining[i]}）")

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

    # 去重（防御性）并截断以满足目标容量（极端情况下）
    def _unique_truncate(ids, cap):
        seen = set()
        result = []
        for pid in ids:
            if pid not in seen:
                seen.add(pid)
                result.append(pid)
            if len(result) >= cap:
                break
        return result

    group1_patients = _unique_truncate(group1_patients, target_group_sizes[0])
    group2_patients = _unique_truncate(group2_patients, target_group_sizes[1])
    group3_patients = _unique_truncate(group3_patients, target_group_sizes[2])

    # 汇总与校验
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

    # 创建输出目录（避免覆盖原有输出）
    output_dir = '../datawithser'
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
    stratify_patients_by_severity_biased_433()


