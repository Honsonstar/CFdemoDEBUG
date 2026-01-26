#!/usr/bin/env python3
"""
真正的嵌套CPCG实现
为每折独立筛选基因，不使用全局CPCG结果
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from scipy.stats import pearsonr, ttest_ind
import warnings
warnings.filterwarnings('ignore')

os.chdir('/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy')

def load_raw_data(study):
    """加载原始CPCG数据"""
    clinical_file = f'preprocessing/CPCG_algo/raw_data/tcga_{study}/clinical.CSV'
    exp_file = f'preprocessing/CPCG_algo/raw_data/tcga_{study}/data.csv'

    if not os.path.exists(clinical_file) or not os.path.exists(exp_file):
        print(f"❌ 原始数据不存在")
        return None, None

    print(f"✅ 加载原始数据")
    clinical = pd.read_csv(clinical_file)
    exp = pd.read_csv(exp_file)

    print(f"   临床数据: {clinical.shape}")
    print(f"   表达数据: {exp.shape}")

    # 转置表达数据（样本 x 基因）
    gene_names = exp['gene_name'].values
    sample_ids = exp.columns[1:].tolist()
    expression_matrix = exp.iloc[:, 1:].values

    exp_df = pd.DataFrame(
        expression_matrix.T,
        index=sample_ids,
        columns=gene_names
    )

    return clinical, exp_df

def survival_gene_selection(clinical, exp_df, train_ids, n_genes=50):
    """
    基于生存分析的基因筛选
    模拟CPCG的核心思想：logrank test + 相关性
    """
    print(f"   开始生存基因筛选...")
    print(f"   训练样本数: {len(train_ids)}")

    # 筛选训练集数据
    train_clinical = clinical[clinical['case_submitter_id'].isin(train_ids)].copy()
    train_exp = exp_df.loc[exp_df.index.intersection(train_ids)].copy()

    # 找到重叠样本
    common_samples = train_clinical['case_submitter_id'].isin(train_exp.index)
    train_clinical = train_clinical[common_samples]
    train_exp = train_exp.loc[train_clinical['case_submitter_id']]

    print(f"   重叠样本数: {len(train_clinical)}")

    if len(train_clinical) < 50:
        print(f"   ⚠️  样本数过少，选择常见基因")
        # 返回一些常见癌症基因作为备用
        common_cancer_genes = [
            'TP53', 'BRCA1', 'BRCA2', 'EGFR', 'MYC', 'RB1', 'PIK3CA', 'KRAS',
            'PTEN', 'APC', 'VHL', 'CDKN2A', 'SMAD4', 'TGFBR2', 'MLH1', 'MSH2',
            'ATM', 'CHEK2', 'PALB2', 'CDH1'
        ]
        available_genes = [g for g in common_cancer_genes if g in train_exp.columns]
        return available_genes[:n_genes]

    # 方法1：基于logrank test的筛选
    gene_scores = []

    for gene in train_exp.columns:
        try:
            # 获取基因表达值
            gene_values = train_exp[gene].astype(float)
            os_times = train_clinical.set_index('case_submitter_id')['OS']
            event_indicators = train_clinical.set_index('case_submitter_id')['Censor']

            # 找到重叠样本
            common_idx = gene_values.index.intersection(os_times.index)
            if len(common_idx) < 30:
                continue

            gene_clean = gene_values.loc[common_idx]
            os_clean = os_times.loc[common_idx]
            event_clean = event_indicators.loc[common_idx]

            # 按中位数分组
            median_val = gene_clean.median()
            low_group = event_clean[gene_clean <= median_val]
            high_group = event_clean[gene_clean > median_val]

            if len(low_group) < 10 or len(high_group) < 10:
                continue

            # 计算两组的中位生存时间
            low_survival = os_clean[gene_clean <= median_val]
            high_survival = os_clean[gene_clean > median_val]

            # 简单的统计检验（这里用t检验代替复杂的logrank test）
            from scipy.stats import ttest_ind
            try:
                stat, p_value = ttest_ind(low_survival, high_survival)
                if not np.isnan(p_value):
                    score = -np.log10(p_value + 1e-10)  # 转换为正值，越大越好
                    gene_scores.append((gene, score, len(common_idx)))
            except:
                continue

        except Exception as e:
            continue

    # 如果logrank test没有找到足够基因，使用相关性方法补充
    if len(gene_scores) < n_genes // 2:
        print(f"   使用相关性方法补充基因...")
        correlations = []
        for gene in train_exp.columns[:1000]:  # 限制基因数量加速计算
            try:
                gene_values = train_exp[gene].astype(float)
                os_times = train_clinical.set_index('case_submitter_id')['OS']

                common_idx = gene_values.index.intersection(os_times.index)
                if len(common_idx) < 30:
                    continue

                gene_clean = gene_values.loc[common_idx]
                os_clean = os_times.loc[common_idx]

                corr, p_val = pearsonr(gene_clean, os_clean)
                if not np.isnan(corr):
                    score = abs(corr) * (-np.log10(p_val + 1e-10))
                    correlations.append((gene, score, len(common_idx)))
            except:
                continue

        # 合并两种方法的结果
        all_genes = gene_scores + correlations

    else:
        all_genes = gene_scores

    # 按分数排序
    all_genes.sort(key=lambda x: x[1], reverse=True)

    # 选择top genes
    selected_genes = [gene for gene, score, n in all_genes[:n_genes]]

    print(f"   ✅ 筛选出 {len(selected_genes)} 个基因")
    if len(selected_genes) > 0:
        print(f"   前10个基因: {selected_genes[:10]}")

    return selected_genes

def create_features_for_fold(global_exp_df, selected_genes, all_ids, clinical):
    """为指定折创建特征文件"""
    feature_data = []

    for sample_id in all_ids:
        if sample_id in global_exp_df.index:
            row = {'sample_id': sample_id}

            # 添加OS
            os_val = clinical[clinical['case_submitter_id'] == sample_id]['OS'].values
            row['OS'] = os_val[0] if len(os_val) > 0 else np.nan

            # 添加基因表达
            for gene in selected_genes:
                if gene in global_exp_df.columns:
                    row[gene] = global_exp_df.loc[sample_id, gene]
                else:
                    row[gene] = 0  # 基因不存在时填充0

            feature_data.append(row)

    return pd.DataFrame(feature_data)

def run_true_nested_cpcg(study, n_genes=50):
    """运行真正的嵌套CPCG"""
    print(f"\n{'='*80}")
    print(f"运行真正嵌套CPCG: {study}")
    print(f"{'='*80}")
    print(f"每折筛选基因数: {n_genes}")

    # 1. 加载原始数据
    clinical, exp_df = load_raw_data(study)
    if clinical is None:
        return None

    # 2. 创建嵌套划分
    clinical_file = f'datasets_csv/clinical_data/tcga_{study}_clinical.csv'
    splits_dir = f'splits/nested_cv/{study}'
    os.makedirs(splits_dir, exist_ok=True)

    print(f"\n📊 创建嵌套划分...")
    clinical_meta = pd.read_csv(clinical_file)
    clinical_meta = clinical_meta.dropna(subset=['case_id', 'censorship'])

    # 匹配样本ID
    sample_ids = clinical_meta['case_id'].values
    labels = clinical_meta['censorship'].values

    print(f"   有效样本数: {len(sample_ids)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_splits = []
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(sample_ids, labels)):
        train_val_ids = sample_ids[train_val_idx]
        test_ids = sample_ids[test_idx]
        train_val_labels = labels[train_val_idx]

        # 划分训练/验证
        train_idx, val_idx = train_test_split(
            np.arange(len(train_val_ids)),
            test_size=0.15,
            stratify=train_val_labels,
            random_state=42
        )

        train_ids = train_val_ids[train_idx]
        val_ids = train_val_ids[val_idx]

        # 保存划分
        max_len = max(len(train_ids), len(val_ids), len(test_ids))
        split_df = pd.DataFrame({
            'train': list(train_ids) + [''] * (max_len - len(train_ids)),
            'val': list(val_ids) + [''] * (max_len - len(val_ids)),
            'test': list(test_ids) + [''] * (max_len - len(test_ids))
        })

        split_file = f'{splits_dir}/nested_splits_{fold}.csv'
        split_df.to_csv(split_file, index=False)

        fold_splits.append({
            'fold': fold,
            'train_ids': train_ids,
            'val_ids': val_ids,
            'test_ids': test_ids
        })

        print(f"   Fold {fold}: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # 3. 为每折独立筛选基因
    print(f"\n🧬 开始每折独立基因筛选...")
    print(f"{'='*80}")

    all_selected_genes = []
    features_dir = f'features/{study}'
    os.makedirs(features_dir, exist_ok=True)

    for fold_idx in range(5):
        fold_info = fold_splits[fold_idx]
        train_ids = fold_info['train_ids']

        print(f"\n{'='*70}")
        print(f"Fold {fold_idx}: 独立基因筛选")
        print(f"{'='*70}")

        # 在训练集上独立筛选基因
        selected_genes = survival_gene_selection(
            clinical, exp_df, train_ids, n_genes
        )

        all_selected_genes.append(selected_genes)

        # 创建特征文件
        all_ids = np.concatenate([
            fold_info['train_ids'],
            fold_info['val_ids'],
            fold_info['test_ids']
        ])

        feature_df = create_features_for_fold(
            exp_df, selected_genes, all_ids, clinical
        )

        # 保存特征文件
        feature_file = f'{features_dir}/fold_{fold_idx}_features.csv'
        feature_df.to_csv(feature_file, index=False)

        print(f"   ✅ 特征文件保存: {feature_file}")
        print(f"   样本数: {len(feature_df)}")
        print(f"   基因数: {len(selected_genes)}")

    # 4. 分析基因差异
    print(f"\n📊 基因筛选结果分析:")
    print(f"{'='*80}")

    print(f"\n各折选择的基因:")
    for i, genes in enumerate(all_selected_genes):
        print(f"   Fold {i}: {len(genes)} 个基因")
        if len(genes) > 0:
            print(f"      前5个: {genes[:5]}")

    # 计算基因重叠
    print(f"\n基因重叠分析:")
    for i in range(5):
        for j in range(i+1, 5):
            overlap = set(all_selected_genes[i]) & set(all_selected_genes[j])
            overlap_rate = len(overlap) / min(len(all_selected_genes[i]), len(all_selected_genes[j])) * 100
            print(f"   Fold {i} vs Fold {j}: {len(overlap)} 个重叠 ({overlap_rate:.1f}%)")

    # 保存汇总
    summary_df = pd.DataFrame([
        {
            'fold': i,
            'n_genes': len(genes),
            'sample_size': fold_splits[i]['train_size'],
            'top_genes': ', '.join(genes[:5]) if genes else ''
        }
        for i, genes in enumerate(all_selected_genes)
    ])
    summary_df.to_csv(f'{features_dir}/gene_selection_summary.csv', index=False)

    print(f"\n{'='*80}")
    print(f"✅ 真正嵌套CPCG完成!")
    print(f"{'='*80}")
    print(f"\n📁 输出目录: {features_dir}")
    print(f"   - fold_0_features.csv 到 fold_4_features.csv")
    print(f"   - gene_selection_summary.csv")

    print(f"\n🎯 接下来:")
    print(f"   1. 使用这些特征文件训练原始深度学习模型")
    print(f"   2. 对比与全局CPCG的性能差异")

    return splits_dir, features_dir, all_selected_genes

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python run_true_nested_cpcg.py <study> [n_genes]")
        print("示例: python run_true_nested_cpcg.py brca 50")
        sys.exit(1)

    study = sys.argv[1]
    n_genes = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    splits_dir, features_dir, selected_genes = run_true_nested_cpcg(study, n_genes)
