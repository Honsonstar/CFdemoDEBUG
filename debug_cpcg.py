#!/usr/bin/env python3
"""调试 CPCG Stage 1 筛选问题"""

import sys
sys.path.insert(0, 'preprocessing/CPCG_algo')

from nested_cv_wrapper import NestedCVFeatureSelector
from Stage1_parametric_model.screen import screen_step_1
import pandas as pd
import os

# 设置参数
study = 'stad'
fold = 0

print("="*60)
print("调试 CPCG Stage 1 筛选")
print("="*60)

# 读取划分文件
split_file = f'splits/nested_cv/{study}/nested_splits_{fold}.csv'
split_df = pd.read_csv(split_file)
train_ids = split_df['train'].dropna().tolist()
train_ids = [str(i)[:12] for i in train_ids]

print(f"\n1. 读取数据...")
# 读取临床数据
clinical_file = f'datasets_csv/clinical_data/tcga_{study}_clinical.csv'
clinical_data = pd.read_csv(clinical_file)
clinical_data['case_id_truncated'] = clinical_data['case_id'].str[:12]
clinical_data['Censor'] = clinical_data['censorship']
clinical_data['OS'] = clinical_data['survival_months']

# 匹配训练集
train_mask = clinical_data['case_id_truncated'].isin(train_ids)
train_clinical = clinical_data[train_mask].copy()

print(f"   训练集样本数: {len(train_clinical)}")
print(f"   死亡样本数: {(train_clinical['Censor']==1).sum()}")
print(f"   删失样本数: {(train_clinical['Censor']==0).sum()}")

# 只使用死亡样本 (Stage 1 parametric 需要)
deaths_clinical = train_clinical[train_clinical['Censor']==1].copy()
print(f"\n2. Stage 1 Parametric 使用死亡样本: {len(deaths_clinical)}")

# 准备数据 (添加 case_submitter_id, Censor, OS)
deaths_clinical['case_submitter_id'] = deaths_clinical['case_id_truncated']
deaths_clinical['Censor'] = deaths_clinical['censorship']
deaths_clinical['OS'] = deaths_clinical['survival_months']
print(f"   添加了必要列: case_submitter_id, Censor, OS")

# 读取表达数据
exp_file = f'datasets_csv/raw_rna_data/combine/{study}/rna_clean.csv'
exp_data = pd.read_csv(exp_file)
exp_data.columns = ['gene_name'] + [c[:12] for c in exp_data.columns[1:]]
exp_data = exp_data.set_index('gene_name')
print(f"\n3. 表达数据基因数: {len(exp_data)}")
print(f"   表达数据样本数: {len(exp_data.columns)}")

# 准备表达数据
exp_reset = exp_data.copy().reset_index()
print(f"\n4. 运行 Stage 1 Parametric 筛选...")

# 调用 Stage 1
result = screen_step_1(
    clinical_final=deaths_clinical,
    exp_data=exp_reset,
    h_type='OS',
    threshold=100,
    n_jobs=4
)

print(f"\n5. Stage 1 结果:")
print(f"   返回类型: {type(result)}")
if isinstance(result, pd.DataFrame):
    print(f"   列名: {list(result.columns)[:10]}...")
    print(f"   行数: {len(result)}")
    genes = [c for c in result.columns if c != 'OS']
    print(f"   筛选出的基因数: {len(genes)}")
    if len(genes) > 0:
        print(f"   前5个基因: {genes[:5]}")
else:
    print(f"   结果: {result}")

print("\n" + "="*60)
