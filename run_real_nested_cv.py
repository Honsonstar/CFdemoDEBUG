#!/usr/bin/env python3
"""
真正的嵌套CV实现
修复CPCG筛选问题，在每折训练数据上独立筛选特征
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_cpog_data(study):
    """加载CPCG原始数据"""
    data_dir = f'preprocessing/CPCG_algo/raw_data/tcga_{study}'

    clinical_file = os.path.join(data_dir, 'clinical.CSV')
    exp_file = os.path.join(data_dir, 'data.csv')

    if not os.path.exists(clinical_file):
        raise FileNotFoundError(f"找不到临床文件: {clinical_file}")
    if not os.path.exists(exp_file):
        raise FileNotFoundError(f"找不到表达文件: {exp_file}")

    # 读取数据
    clinical = pd.read_csv(clinical_file)
    exp = pd.read_csv(exp_file)

    print(f"✅ 加载数据成功:")
    print(f"   临床数据: {clinical.shape}")
    print(f"   表达数据: {exp.shape}")

    return clinical, exp

def prepare_expression_data(exp):
    """准备表达数据：转置使其样本为行，基因为列"""
    gene_names = exp['gene_name'].values
    sample_ids = exp.columns[1:].tolist()
    expression_matrix = exp.iloc[:, 1:].values

    exp_df = pd.DataFrame(
        expression_matrix.T,
        index=sample_ids,
        columns=gene_names
    )

    return exp_df

def simple_gene_selection(clinical, exp_df, train_ids, n_genes=100):
    """简化的基因筛选：使用相关性筛选与生存相关的基因"""
    print(f"   开始基因筛选...")
    print(f"   训练样本数: {len(train_ids)}")

    # 获取训练集数据
    train_clinical = clinical[clinical['case_submitter_id'].isin(train_ids)].copy()

    # 找到重叠样本
    common_samples = train_clinical['case_submitter_id'].isin(exp_df.index)
    train_clinical = train_clinical[common_samples]

    print(f"   重叠样本数: {len(train_clinical)}")

    # 如果样本太少，使用全局CPCG结果
    if len(train_clinical) < 50:
        print(f"   ⚠️  样本数过少，使用全局CPCG结果")
        return None  # 返回None表示使用全局结果

    # 计算基因与OS的相关性
    correlations = []
    for gene in exp_df.columns:
        try:
            gene_vals = exp_df.loc[train_clinical['case_submitter_id'], gene].astype(float)
            os_vals = train_clinical.set_index('case_submitter_id')['OS']

            # 移除缺失值
            mask = ~(gene_vals.isna() | os_vals.isna())
            if mask.sum() < 20:
                continue

            gene_clean = gene_vals[mask]
            os_clean = os_vals[mask]

            # 计算Pearson相关系数
            corr = np.corrcoef(gene_clean, os_clean)[0, 1]
            if not np.isnan(corr):
                correlations.append((gene, abs(corr)))
        except Exception as e:
            continue

    # 按相关性排序
    correlations.sort(key=lambda x: x[1], reverse=True)

    # 选择top genes
    selected_genes = [gene for gene, corr in correlations[:n_genes]]

    print(f"   ✅ 筛选出 {len(selected_genes)} 个基因")

    return selected_genes

def run_nested_cv(study):
    """运行嵌套交叉验证"""

    print(f"\n{'='*80}")
    print(f"运行嵌套交叉验证 (修改后方案): {study}")
    print(f"{'='*80}")

    # 1. 加载数据
    clinical, exp = load_cpog_data(study)
    exp_df = prepare_expression_data(exp)

    # 2. 创建嵌套CV划分
    splits_dir = f'splits/nested_cv/{study}'
    os.makedirs(splits_dir, exist_ok=True)

    # 获取样本和标签
    valid_clinical = clinical.dropna(subset=['case_submitter_id', 'censorship'])
    sample_ids = valid_clinical['case_submitter_id'].values
    labels = valid_clinical['censorship'].values

    print(f"\n📊 创建5折交叉验证:")
    print(f"   有效样本数: {len(sample_ids)}")

    # 5折交叉验证
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
            'test_ids': test_ids,
            'train_size': len(train_ids),
            'val_size': len(val_ids),
            'test_size': len(test_ids)
        })

        print(f"   Fold {fold}: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # 3. 为每折筛选特征
    print(f"\n🧬 开始每折独立特征筛选...")
    print(f"{'='*80}")

    features_dir = f'features/{study}'
    os.makedirs(features_dir, exist_ok=True)

    # 读取全局CPCG结果作为备用
    global_file = f'preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_{study}/{study}_M2M3base_0916.csv'
    global_genes = None
    if os.path.exists(global_file):
        global_cpcg = pd.read_csv(global_file, index_col=0)
        global_genes = [col for col in global_cpcg.columns if col != 'OS']
        print(f"📦 加载全局CPCG结果: {len(global_genes)} 个基因")

    for fold_info in fold_splits:
        fold = fold_info['fold']
        train_ids = fold_info['train_ids']

        print(f"\nFold {fold}:")
        print(f"  {'-'*60}")

        # 在训练集上筛选特征
        selected_genes = simple_gene_selection(
            clinical, exp_df, train_ids, n_genes=100
        )

        # 如果筛选失败，使用全局基因
        if selected_genes is None or len(selected_genes) == 0:
            selected_genes = global_genes[:100] if global_genes else []
            print(f"  使用全局CPCG前100个基因")

        # 生成特征文件
        all_ids = np.concatenate([
            fold_info['train_ids'],
            fold_info['val_ids'],
            fold_info['test_ids']
        ])

        feature_data = []
        for sample_id in all_ids:
            if sample_id in exp_df.index:
                row = {'sample_id': sample_id}
                # 添加OS
                os_val = clinical[clinical['case_submitter_id'] == sample_id]['OS'].values
                row['OS'] = os_val[0] if len(os_val) > 0 else np.nan
                # 添加基因表达
                for gene in selected_genes:
                    if gene in exp_df.columns:
                        row[gene] = exp_df.loc[sample_id, gene]
                feature_data.append(row)

        # 保存特征文件
        feature_df = pd.DataFrame(feature_data)
        feature_file = f'{features_dir}/fold_{fold}_features.csv'
        feature_df.to_csv(feature_file, index=False)

        print(f"  ✅ 特征文件: {feature_file}")
        print(f"  样本数: {len(feature_df)}")
        print(f"  基因数: {len(selected_genes)}")

    # 4. 保存汇总
    summary = pd.DataFrame([
        {
            'fold': f['fold'],
            'train_size': f['train_size'],
            'val_size': f['val_size'],
            'test_size': f['test_size'],
            'feature_file': f'features/{study}/fold_{f["fold"]}_features.csv'
        }
        for f in fold_splits
    ])
    summary.to_csv(f'{features_dir}/summary.csv', index=False)

    print(f"\n{'='*80}")
    print(f"✅ 嵌套CV特征筛选完成!")
    print(f"{'='*80}")
    print(f"\n📁 输出目录:")
    print(f"   划分文件: {splits_dir}/")
    print(f"   特征文件: {features_dir}/")

    print(f"\n📊 各折基因筛选结果:")
    for fold in range(5):
        feature_file = f'{features_dir}/fold_{fold}_features.csv'
        if os.path.exists(feature_file):
            df = pd.read_csv(feature_file)
            n_genes = len([col for col in df.columns if col not in ['sample_id', 'OS']])
            print(f"   Fold {fold}: {n_genes} 基因, {len(df)} 样本")

    print(f"\n🎯 接下来运行训练:")
    print(f"   python main_nested.py --study tcga_{study} --ab_model 2 \\")
    print(f"       --split_dir {splits_dir} \\")
    print(f"       --features_dir {features_dir} \\")
    print(f"       --results_dir results_nested_cv_{study}")

    return features_dir, splits_dir

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python run_real_nested_cv.py <study>")
        print("示例: python run_real_nested_cv.py brca")
        sys.exit(1)

    study = sys.argv[1]
    features_dir, splits_dir = run_nested_cv(study)
