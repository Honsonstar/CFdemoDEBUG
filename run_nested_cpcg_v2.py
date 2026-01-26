#!/usr/bin/env python3
"""
简化的嵌套CPCG实现
使用全局CPCG结果作为特征池，每折独立选择子集
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import warnings
warnings.filterwarnings('ignore')

os.chdir('/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy')

def load_global_cpcg(study):
    """加载全局CPCG结果作为候选特征"""
    global_file = f'preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_{study}/tcga_{study}_M2M3base_0916.csv'

    if not os.path.exists(global_file):
        print(f"❌ 全局CPCG结果不存在: {global_file}")
        return None, None

    print(f"✅ 加载全局CPCG: {global_file}")
    global_cpcg = pd.read_csv(global_file, index_col=0)

    # 获取基因和OS
    gene_cols = [col for col in global_cpcg.columns if col != 'OS']
    os_times = global_cpcg['OS'].values

    print(f"   全局CPCG基因数: {len(gene_cols)}")
    print(f"   样本数: {len(global_cpcg)}")

    return global_cpcg, gene_cols

def create_nested_splits(study):
    """创建嵌套CV划分"""
    clinical_file = f'datasets_csv/clinical_data/tcga_{study}_clinical.csv'
    splits_dir = f'splits/nested_cv/{study}'

    os.makedirs(splits_dir, exist_ok=True)

    print(f"\n📊 创建嵌套划分...")
    clinical = pd.read_csv(clinical_file)
    clinical = clinical.dropna(subset=['case_id', 'censorship'])

    sample_ids = clinical['case_id'].values
    labels = clinical['censorship'].values

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
            'test_ids': test_ids,
            'train_size': len(train_ids),
            'val_size': len(val_ids),
            'test_size': len(test_ids)
        })

        print(f"   Fold {fold}: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    return splits_dir, fold_splits

def select_features_for_fold(global_cpcg, gene_cols, fold_splits, fold_idx, n_genes=50):
    """为指定折在训练集上选择特征"""
    fold_info = fold_splits[fold_idx]
    train_ids = fold_info['train_ids']

    print(f"\n{'='*70}")
    print(f"Fold {fold_idx}: 特征选择")
    print(f"{'='*70}")
    print(f"   训练样本数: {len(train_ids)}")
    print(f"   候选基因数: {len(gene_cols)}")

    # 获取训练集数据
    train_mask = global_cpcg.index.isin(train_ids)
    X_train_all = global_cpcg.loc[train_mask, gene_cols].fillna(0).values
    y_train = global_cpcg.loc[train_mask, 'OS'].values

    print(f"   训练数据形状: {X_train_all.shape}")

    # 使用F统计量选择特征
    # 注意：这是简化的选择，实际CPCG使用更复杂的方法
    selector = SelectKBest(score_func=f_regression, k=min(n_genes, X_train_all.shape[1]))
    X_train_selected = selector.fit_transform(X_train_all, y_train)

    # 获取选中特征的索引
    selected_indices = selector.get_support(indices=True)
    selected_genes = [gene_cols[i] for i in selected_indices]

    print(f"   选中基因数: {len(selected_genes)}")
    print(f"   前10个基因: {selected_genes[:10]}")

    return selected_genes

def train_fold_nested(global_cpcg, gene_cols, fold_splits, fold_idx, selected_genes):
    """使用选中特征训练模型"""
    fold_info = fold_splits[fold_idx]
    train_ids = fold_info['train_ids']
    val_ids = fold_info['val_ids']
    test_ids = fold_info['test_ids']

    print(f"\n{'='*70}")
    print(f"Fold {fold_idx}: 训练模型")
    print(f"{'='*70}")

    # 准备数据
    train_mask = global_cpcg.index.isin(train_ids)
    val_mask = global_cpcg.index.isin(val_ids)
    test_mask = global_cpcg.index.isin(test_ids)

    X_train = global_cpcg.loc[train_mask, selected_genes].fillna(0).values
    y_train_time = global_cpcg.loc[train_mask, 'OS'].values
    y_train = np.array(
        [(True, t) for t in y_train_time],
        dtype=[('event', bool), ('time', float)]
    )

    X_test = global_cpcg.loc[test_mask, selected_genes].fillna(0).values
    y_test_time = global_cpcg.loc[test_mask, 'OS'].values
    y_test = np.array(
        [(True, t) for t in y_test_time],
        dtype=[('event', bool), ('time', float)]
    )

    print(f"   训练集: {X_train.shape}")
    print(f"   测试集: {X_test.shape}")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    print(f"   训练Cox模型...")
    model = CoxnetSurvivalAnalysis(
        l1_ratio=0.5,
        max_iter=1000
    )

    model.fit(X_train_scaled, y_train)

    # 预测和评估
    y_pred = model.predict(X_test_scaled)
    c_index = concordance_index_censored(
        y_test['event'], y_test['time'], y_pred
    )[0]

    print(f"   ✅ C-index: {c_index:.4f}")

    return {
        'fold': fold_idx,
        'c_index': c_index,
        'train_size': len(train_ids),
        'test_size': len(test_ids),
        'n_genes': len(selected_genes)
    }

def run_nested_cpcg(study, n_genes=50):
    """运行嵌套CPCG"""
    print(f"\n{'='*80}")
    print(f"运行嵌套CPCG (简化版): {study}")
    print(f"{'='*80}")
    print(f"每折选择基因数: {n_genes}")

    # 1. 加载全局CPCG结果
    global_cpcg, gene_cols = load_global_cpcg(study)
    if global_cpcg is None:
        return None

    # 2. 创建嵌套划分
    splits_dir, fold_splits = create_nested_splits(study)

    # 3. 为每折独立选择特征并训练
    print(f"\n🧬 开始每折独立特征选择和训练...")
    print(f"{'='*80}")

    all_results = []

    for fold_idx in range(5):
        # 选择特征（仅在训练集上）
        selected_genes = select_features_for_fold(
            global_cpcg, gene_cols, fold_splits, fold_idx, n_genes
        )

        # 训练模型
        result = train_fold_nested(
            global_cpcg, gene_cols, fold_splits, fold_idx, selected_genes
        )

        all_results.append(result)

    # 4. 汇总结果
    print(f"\n{'='*80}")
    print(f"📊 嵌套CPCG结果汇总")
    print(f"{'='*80}")

    results_df = pd.DataFrame(all_results)
    results_dir = f'results/nested_cpcg_{study}'
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(f'{results_dir}/summary.csv', index=False)

    mean_cindex = np.mean([r['c_index'] for r in all_results])
    std_cindex = np.std([r['c_index'] for r in all_results])

    print(f"\n🎯 最终结果 (嵌套CPCG):")
    print(f"{'='*80}")
    print(f"C-index: {mean_cindex:.4f} ± {std_cindex:.4f}")
    print(f"\n各折详情:")
    for result in all_results:
        print(f"   Fold {result['fold']}: {result['c_index']:.4f} "
              f"(Train={result['train_size']}, Test={result['test_size']}, Genes={result['n_genes']})")

    print(f"\n💾 结果保存到: {results_dir}/summary.csv")

    return mean_cindex, std_cindex

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python run_nested_cpcg_v2.py <study> [n_genes]")
        print("示例: python run_nested_cpcg_v2.py brca 50")
        sys.exit(1)

    study = sys.argv[1]
    n_genes = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    mean_cindex, std_cindex = run_nested_cpcg(study, n_genes)

    if mean_cindex is not None:
        print(f"\n{'='*80}")
        print(f"✅ 嵌套CPCG完成!")
        print(f"{'='*80}")
