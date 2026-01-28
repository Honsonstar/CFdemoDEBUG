#!/usr/bin/env python3
"""调试 CPCG Stage 1 筛选问题 - 详细版"""

import sys
sys.path.insert(0, 'preprocessing/CPCG_algo')

import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test
from pingouin import partial_corr
import warnings
warnings.filterwarnings('ignore')

# 设置参数
study = 'stad'

print("="*60)
print("调试 CPCG Stage 1 筛选 - 详细日志")
print("="*60)

# 读取划分文件
split_file = f'splits/nested_cv/{study}/nested_splits_0.csv'
split_df = pd.read_csv(split_file)
train_ids = split_df['train'].dropna().tolist()
train_ids = [str(i)[:12] for i in train_ids]

# 读取临床数据
clinical_file = f'datasets_csv/clinical_data/tcga_{study}_clinical.csv'
clinical_data = pd.read_csv(clinical_file)
clinical_data['case_id_truncated'] = clinical_data['case_id'].str[:12]
clinical_data['Censor'] = clinical_data['censorship']
clinical_data['OS'] = clinical_data['survival_months']
clinical_data['case_submitter_id'] = clinical_data['case_id_truncated']

# 匹配训练集
train_mask = clinical_data['case_id_truncated'].isin(train_ids)
train_clinical = clinical_data[train_mask].copy()

# 只使用死亡样本 (Stage 1 parametric 需要)
deaths_clinical = train_clinical[train_clinical['Censor']==1].copy()

print(f"\n训练集样本数: {len(train_clinical)}")
print(f"死亡样本数: {len(deaths_clinical)}")

# 读取表达数据
exp_file = f'datasets_csv/raw_rna_data/combine/{study}/rna_clean.csv'
exp_data = pd.read_csv(exp_file)
exp_data.columns = ['gene_name'] + [c[:12] for c in exp_data.columns[1:]]
exp_data = exp_data.set_index('gene_name')
exp_reset = exp_data.copy().reset_index()

print(f"\n基因数: {len(exp_data)}")
print(f"样本数: {len(exp_data.columns)}")

# 测试前 5 个基因
print("\n" + "="*60)
print("测试前 5 个基因的筛选结果")
print("="*60)

gene_names = exp_data.index[:5].tolist()

for gene_name in gene_names:
    print(f"\n测试基因: {gene_name}")

    # 准备基因数据
    gene_data = exp_data.loc[[gene_name]]
    temp_data = gene_data.T.copy()
    temp_data.columns = [gene_name]

    # 合并数据
    cd_copy = deaths_clinical.copy()
    cd_copy = cd_copy.set_index('case_submitter_id')
    cd_copy = cd_copy.merge(temp_data, how='left', left_index=True, right_index=True)
    cd_copy = cd_copy.reset_index()

    # 检查数据类型
    try:
        cd_copy[gene_name] = cd_copy[gene_name].astype(float)
    except Exception as e:
        print(f"   ❌ 类型转换失败: {e}")
        continue

    # 检查缺失值
    cd_copy = cd_copy.dropna(subset=[gene_name, 'OS', 'Censor'])
    print(f"   合并后样本数: {len(cd_copy)}")

    if len(cd_copy) == 0:
        print(f"   ❌ 跳过: 无有效样本")
        continue

    # 中位数分组
    median_val = cd_copy[gene_name].median()
    d_l = cd_copy[cd_copy[gene_name] <= median_val].copy()
    d_h = cd_copy[cd_copy[gene_name] > median_val].copy()

    print(f"   低表达组: {len(d_l)} 样本")
    print(f"   高表达组: {len(d_h)} 样本")

    # 检查分组样本数
    if len(d_l) < 6 or len(d_h) < 6:
        print(f"   ❌ 跳过: 分组样本数不足 (需要各组 >= 6)")
        continue

    # Logrank test
    try:
        results = logrank_test(d_l['OS'], d_h['OS'], d_l['Censor'], d_h['Censor'])
        p_value = results.p_value
        print(f"   Logrank p-value: {p_value:.6f}")
    except Exception as e:
        print(f"   ❌ Logrank test 失败: {e}")
        continue

    if p_value > 0.01:
        print(f"   ❌ 跳过: p-value ({p_value:.6f}) > 0.01")
        continue

    # 偏相关分析
    deaths_only = cd_copy[cd_copy['Censor']==1]
    if len(deaths_only) < 6:
        print(f"   ❌ 跳过: 死亡样本不足 ({len(deaths_only)})")
        continue

    try:
        corr_pd = partial_corr(data=deaths_only, x=gene_name, y='OS')
        if corr_pd is not None and 'pearson' in corr_pd.index and 'r' in corr_pd.columns:
            corr_value = np.abs(corr_pd.loc['pearson', 'r'])
            print(f"   ✅ 通过! 偏相关系数: {corr_value:.4f}")
        else:
            print(f"   ❌ 跳过: 偏相关分析失败")
    except Exception as e:
        print(f"   ❌ 跳过: 偏相关分析异常: {e}")

print("\n" + "="*60)
print("调试完成")
print("="*60)
