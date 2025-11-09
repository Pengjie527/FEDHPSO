import pandas as pd
import numpy as np

# ==========================
# 1. 读取数据
# ==========================
df = pd.read_csv("E:\研三\py_ai_clinician-master\Dataset\MIMICtable.csv")

# 关键指标
KEY_COLS = ['SOFA','Arterial_lactate','max_dose_vaso','PaO2_FiO2','GCS']

# ==========================
# 2. 通过循环取每个病人最早记录
# ==========================
first_records = []
seen_icuid = set()

for idx, row in df.iterrows():
    icu_id = row['icustayid']
    if icu_id not in seen_icuid:
        first_records.append(row)
        seen_icuid.add(icu_id)

df_first = pd.DataFrame(first_records).reset_index(drop=True)

# ==========================
# 3. 单指标打分函数（1-轻，2-中，3-重）
# ==========================
def score_sofa(val):
    if pd.isna(val) or val == 0:
        return None
    elif val >= 10:
        return 3
    elif val >= 5:
        return 2
    else:
        return 1

def score_lactate(val):
    if pd.isna(val) or val == 0:
        return None
    elif val > 4:
        return 3
    elif val >= 2:
        return 2
    else:
        return 1

def score_vaso(val):
    if pd.isna(val) or val == 0:
        return None
    else:
        return 3

def score_pafi(val):
    if pd.isna(val) or val == 0:
        return None
    elif val < 100:
        return 3
    elif val < 200:
        return 2
    elif val < 300:
        return 1
    else:
        return 1

def score_gcs(val):
    if pd.isna(val) or val == 0:
        return None
    elif val <= 8:
        return 3
    elif val <= 13:
        return 2
    else:
        return 1

# ==========================
# 4. 计算每个病人的平均分
# ==========================
scores_list = []
for idx, row in df_first.iterrows():
    scores = []
    s = score_sofa(row.get('SOFA'))
    if s is not None:
        scores.append(s)
    s = score_lactate(row.get('Arterial_lactate'))
    if s is not None:
        scores.append(s)
    s = score_vaso(row.get('max_dose_vaso'))
    if s is not None:
        scores.append(s)
    s = score_pafi(row.get('PaO2_FiO2'))
    if s is not None:
        scores.append(s)
    s = score_gcs(row.get('GCS'))
    if s is not None:
        scores.append(s)
    if len(scores) == 0:
        avg_score = np.nan
    else:
        avg_score = np.mean(scores)
    scores_list.append(avg_score)

df_first['avg_score'] = scores_list

# ==========================
# 5. 根据平均分划分轻/中/重
# ==========================
def avg_score_to_severity(score):
    if pd.isna(score):
        return 'Unknown'
    elif score < 1.5:
        return '1'
    elif score < 2.5:
        return '2'
    else:
        return '3'

df_first['severity'] = df_first['avg_score'].apply(avg_score_to_severity)

# ==========================
# 6. 生成病人与病情对照表
# ==========================
severity_table = df_first[['icustayid','severity']].copy()
severity_table.to_csv("severity_table.csv", index=False)

print("每个等级人数统计：")
print(severity_table['severity'].value_counts())
print("映射表已保存为 severity_table.csv")
