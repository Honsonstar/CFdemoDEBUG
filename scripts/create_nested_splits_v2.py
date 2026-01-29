#!/usr/bin/env python3
"""
重新生成所有癌症类型的嵌套交叉验证划分
基于用户提供的模板
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import argparse

def create_splits(study, n_splits=5, seed=42):
    print(f"\n==========================================")
    print(f"🚀 正在处理癌种: {study}")
    print(f"==========================================")

    # 1. 路径设置
    clinical_file = f"datasets_csv/clinical_data/tcga_{study}_clinical.csv"
    output_dir = f"splits/nested_cv/{study}"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(clinical_file):
        print(f"❌ 错误: 找不到文件 {clinical_file}")
        return

    # 2. 读取数据
    df = pd.read_csv(clinical_file)
    print(f"   原始样本数: {len(df)}")

    # 3. 清洗数据
    # 使用 case_id 作为样本ID
    df = df.dropna(subset=['case_id'])
    # 使用 censorship 作为分层标签
    df = df.dropna(subset=['censorship'])

    ids = df['case_id'].values
    labels = df['censorship'].values
    print(f"   有效样本数: {len(ids)}")

    # 4. 5 折交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(ids, labels)):
        print(f"\n🔄 生成 Fold {fold}...")

        train_val_ids = ids[train_val_idx]
        test_ids = ids[test_idx]
        train_val_labels = labels[train_val_idx]

        # 5. 划分 Train/Val (85% / 15% of Train+Val)
        train_idx, val_idx = train_test_split(
            np.arange(len(train_val_ids)),
            test_size=0.15,
            stratify=train_val_labels,
            random_state=seed
        )

        train_ids = train_val_ids[train_idx]
        val_ids = train_val_ids[val_idx]

        print(f"   ✓ Train: {len(train_ids)}")
        print(f"   ✓ Val:   {len(val_ids)}")
        print(f"   ✓ Test:  {len(test_ids)}")

        # 6. 保存 CSV (格式对齐 nested_cv_wrapper.py)
        max_len = max(len(train_ids), len(val_ids), len(test_ids))

        # 填充 NaN 以对齐长度
        train_col = list(train_ids) + [np.nan] * (max_len - len(train_ids))
        val_col = list(val_ids) + [np.nan] * (max_len - len(val_ids))
        test_col = list(test_ids) + [np.nan] * (max_len - len(test_ids))

        split_df = pd.DataFrame({
            'train': train_col,
            'val': val_col,
            'test': test_col
        })

        save_path = os.path.join(output_dir, f"nested_splits_{fold}.csv")
        split_df.to_csv(save_path, index=False)
        print(f"   💾 保存至: {save_path}")

    # 7. 生成汇总文件 (Summary)
    summary_data = []
    for fold in range(n_splits):
        f_path = os.path.join(output_dir, f"nested_splits_{fold}.csv")
        if os.path.exists(f_path):
            d = pd.read_csv(f_path)
            summary_data.append({
                'fold': fold,
                'train': d['train'].notna().sum(),
                'val': d['val'].notna().sum(),
                'test': d['test'].notna().sum()
            })
    pd.DataFrame(summary_data).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"\n✅ {study} 划分完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--studies', nargs='+',
                        default=['blca', 'brca', 'coadread', 'hnsc', 'stad'],
                        help='要处理的癌种列表')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    for study in args.studies:
        create_splits(study, seed=args.seed)
