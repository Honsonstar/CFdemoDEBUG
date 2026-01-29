#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys

if len(sys.argv) < 2:
    print("用法: python3 test_split.py <cancer>")
    sys.exit(1)

STUDY = sys.argv[1]
print(f"\n{'='*60}")
print(f"测试 {STUDY.upper()} 的划分生成")
print(f"{'='*60}")

# 读取数据
CLINICAL_FILE = f'datasets_csv/clinical_data/tcga_{STUDY}_clinical.csv'
df = pd.read_csv(CLINICAL_FILE)
print(f"总样本数: {len(df)}")

# 清理数据
df = df.dropna(subset=['case_id', 'censorship'])
df = df[df['case_id'].astype(bool)]
print(f"清洗后: {len(df)}")

# 获取ID和标签
ids = df['case_id'].values
labels = df['censorship'].values

# 5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n生成5折交叉验证划分:")
train_sets = []
test_sets = []

for fold, (train_val_idx, test_idx) in enumerate(skf.split(ids, labels)):
    train_val_ids = ids[train_val_idx]
    test_ids = ids[test_idx]
    train_val_labels = labels[train_val_idx]

    # 划分 train/val
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_ids)),
        test_size=0.15,
        stratify=train_val_labels,
        random_state=42
    )

    train_ids = train_val_ids[train_idx]
    val_ids = train_val_ids[val_idx]

    train_sets.append(set(train_ids))
    test_sets.append(set(test_idx))

    print(f"Fold {fold}: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

# 检查是否都相同
all_train_same = all(s == train_sets[0] for s in train_sets[1:])
all_test_same = all(s == test_sets[0] for s in test_sets[1:])

if all_train_same:
    print(f"\n❌ 错误: 所有 Fold 训练集相同!")
else:
    print(f"\n✅ 成功: 各 Fold 训练集不同")
    
    # 检查交集
    print(f"\n各 Fold 训练集交集大小:")
    for i in range(5):
        for j in range(i+1, 5):
            common = len(train_sets[i] & train_sets[j])
            print(f"  Fold {i} ∩ Fold {j} = {common}")

if all_test_same:
    print(f"❌ 错误: 所有 Fold 测试集相同!")
else:
    print(f"✅ 成功: 各 Fold 测试集不同")
