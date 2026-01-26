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
print(f"总样本数: {len(df)}")

ids = df['case_id'].values if 'case_id' in df.columns else df.iloc[:, 0].values
labels = df['censorship'].values if 'censorship' in df.columns else df.iloc[:, 1].values

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

    split_df = pd.DataFrame({
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
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
print("\n✅ 完成!")
