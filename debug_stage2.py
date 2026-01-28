#!/usr/bin/env python3
"""调试 Stage 2 数据准备"""

import pandas as pd
import numpy as np

study = 'stad'

print("="*60)
print("调试 Stage 2 数据准备")
print("="*60)

# 1. 读取划分文件
split_file = f'splits/nested_cv/{study}/nested_splits_0.csv'
split_df = pd.read_csv(split_file)
train_ids = split_df['train'].dropna().tolist()
train_ids = [str(i)[:12] for i in train_ids]

print(f'\n1. 划分文件:')
print(f'   Train IDs: {len(train_ids)}')
print(f'   唯一 IDs: {len(set(train_ids))}')
print(f'   前5个: {train_ids[:5]}')

# 2. 读取临床数据
clinical_file = f'datasets_csv/clinical_data/tcga_{study}_clinical.csv'
clin = pd.read_csv(clinical_file)
clin['case_id_truncated'] = clin['case_id'].str[:12]
clin['OS'] = clin['survival_months']
clin['Censor'] = clin['censorship']

print(f'\n2. 临床数据:')
print(f'   总样本: {len(clin)}')
print(f'   唯一 truncated IDs: {clin["case_id_truncated"].nunique()}')

# 3. 筛选训练集
train_mask = clin['case_id_truncated'].isin(train_ids)
train_clinical = clin[train_mask].copy()
print(f'\n3. 训练集临床数据:')
print(f'   匹配样本: {len(train_clinical)}')

# 4. 设置索引
train_clinical_idx = train_clinical.set_index('case_id_truncated')
train_clinical_idx.index = train_clinical_idx.index.astype(str)

# 去除重复索引
if train_clinical_idx.index.duplicated().any():
    print(f'   [去重] 移除 {train_clinical_idx.index.duplicated().sum()} 个重复ID')
    train_clinical_idx = train_clinical_idx[~train_clinical_idx.index.duplicated(keep='first')]

print(f'   去重后样本: {len(train_clinical_idx)}')

# 5. 读取表达数据
exp_file = f'datasets_csv/raw_rna_data/combine/{study}/rna_clean.csv'
exp_data = pd.read_csv(exp_file, index_col=0)
exp_data.index = [str(i)[:12] for i in exp_data.index]

print(f'\n4. 表达数据:')
print(f'   样本数: {len(exp_data)}')
print(f'   基因数: {len(exp_data.columns)}')

# 6. 转置
exp_T = exp_data.T
exp_T.index = exp_T.index.astype(str)

# 7. 模拟 Stage 1 候选基因
# 取前 10 个基因作为测试
candidates = list(exp_T.columns[:10])
print(f'\n5. 候选基因 (测试用): {len(candidates)}')
print(f'   {candidates[:5]}')

# 8. 过滤有效的基因
valid_genes = [g for g in candidates if g in exp_T.columns]
print(f'\n6. 有效基因: {len(valid_genes)}')

# 9. Merge 测试
print(f'\n7. Merge 测试:')
print(f'   train_clinical_idx 索引: {len(train_clinical_idx)} 个')
print(f'   exp_T 索引: {len(exp_T)} 个')
print(f'   共同索引: {len(set(train_clinical_idx.index) & set(exp_T.index))} 个')

merged = pd.merge(
    train_clinical_idx[['OS']],
    exp_T[valid_genes],
    left_index=True,
    right_index=True,
    how='inner'
)

print(f'\n8. Merge 结果:')
print(f'   Shape: {merged.shape}')
print(f'   列名: {list(merged.columns)[:5]}...')
print(f'   样本数: {len(merged)}')
print(f'   基因列数: {len(merged.columns) - 1}')

if merged.shape[1] <= 1:
    print(f'\n❌ 问题: merge 失败，基因列未合并进来')
else:
    print(f'\n✅ 成功: merge 正常')
