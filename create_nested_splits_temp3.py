import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import sys

study = sys.argv[1] if len(sys.argv) > 1 else 'brca'
clinical_file = f'datasets_csv/clinical_data/tcga_{study}_clinical.csv'
output_dir = f'splits/nested_cv/{study}'
os.makedirs(output_dir, exist_ok=True)

print(f"创建嵌套CV划分: {study}")
df = pd.read_csv(clinical_file)
print(f"原始样本数: {len(df)}")

# 过滤掉缺失值
df = df.dropna(subset=['case_id', 'censorship'])
print(f"有效样本数: {len(df)}")

ids = df['case_id'].values
labels = df['censorship'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold in range(5):
    train_val_idx, test_idx = list(skf.split(ids, labels))[fold]
    train_val_ids = ids[train_val_idx]
    test_ids = ids[test_idx]
    train_val_labels = labels[train_val_idx]

    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_ids)),
        test_size=0.15,
        stratify=train_val_labels,
        random_state=42
    )

    train_ids = train_val_ids[train_idx]
    val_ids = train_val_ids[val_idx]

    # Pad columns to same length
    max_len = max(len(train_ids), len(val_ids), len(test_ids))
    train_col = list(train_ids) + [''] * (max_len - len(train_ids))
    val_col = list(val_ids) + [''] * (max_len - len(val_ids))
    test_col = list(test_ids) + [''] * (max_len - len(test_ids))

    split_df = pd.DataFrame({
        'train': train_col,
        'val': val_col,
        'test': test_col
    })

    output_file = f'{output_dir}/nested_splits_{fold}.csv'
    split_df.to_csv(output_file, index=False)

    print(f"Fold {fold}: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

summary_df = pd.DataFrame([{
    'fold': fold,
    'train': len(pd.read_csv(f'{output_dir}/nested_splits_{fold}.csv')['train'].dropna()),
    'val': len(pd.read_csv(f'{output_dir}/nested_splits_{fold}.csv')['val'].dropna()),
    'test': len(pd.read_csv(f'{output_dir}/nested_splits_{fold}.csv')['test'].dropna())
} for fold in range(5)])
summary_df.to_csv(f'{output_dir}/summary.csv', index=False)
print("\n✅ 完成! 输出目录:", output_dir)
