#!/bin/bash
# 快速创建嵌套CV划分的脚本

STUDY=$1
if [ -z "$STUDY" ]; then
    echo "=========================================="
    echo "用法: bash create_nested_splits.sh <study>"
    echo ""
    echo "可用癌种:"
    echo "  - blca (膀胱尿路上皮癌)"
    echo "  - brca (乳腺浸润癌)"
    echo "  - hnsc (头颈鳞状细胞癌)"
    echo "  - stad (胃腺癌)"
    echo "  - coadread (结直肠腺癌)"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "创建嵌套CV划分: $STUDY"
echo "=========================================="

# 检查临床数据文件
CLINICAL_FILE="datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv"
if [ ! -f "$CLINICAL_FILE" ]; then
    echo "❌ 错误: 找不到临床数据文件 $CLINICAL_FILE"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="splits/nested_cv/${STUDY}"
mkdir -p "$OUTPUT_DIR"

# 运行Python脚本创建划分
python3 << PYTHON
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import os

print("\n📊 开始创建嵌套CV划分...")
print(f"   癌种: $STUDY")
print(f"   输入文件: $CLINICAL_FILE")
print(f"   输出目录: $OUTPUT_DIR")

# 读取数据
df = pd.read_csv('$CLINICAL_FILE')
print(f"   总样本数: {len(df)}")

# 清理数据
df = df.dropna(subset=['case_id', 'censorship'])
df = df[df['case_id'].astype(bool)]
print(f"   清洗后样本数: {len(df)}")

# 获取ID和标签
ids = df['case_id'].values
labels = df['censorship'].values

print(f"   有效样本数: {len(ids)}")

# 5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold in range(5):
    print(f"\n🔄 处理 Fold {fold}...")

    train_val_idx, test_idx = next(skf.split(ids, labels))
    train_val_ids = ids[train_val_idx]
    test_ids = ids[test_idx]
    train_val_labels = labels[train_val_idx]

    # 划分训练/验证 (85% / 15%)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_ids)),
        test_size=0.15,
        stratify=train_val_labels,
        random_state=42
    )

    train_ids = train_val_ids[train_idx]
    val_ids = train_val_ids[val_idx]

    print(f"   ✓ Train: {len(train_ids):3d} 样本")
    print(f"   ✓ Val:   {len(val_ids):3d} 样本")
    print(f"   ✓ Test:  {len(test_ids):3d} 样本")

    # 直接保存为CSV，不使用DataFrame
    output_file = f'$OUTPUT_DIR/nested_splits_{fold}.csv'
    with open(output_file, 'w') as f:
        f.write('train,val,test\n')
        max_len = max(len(train_ids), len(val_ids), len(test_ids))
        for i in range(max_len):
            train = str(train_ids[i]) if i < len(train_ids) else ''
            val = str(val_ids[i]) if i < len(val_ids) else ''
            test = str(test_ids[i]) if i < len(test_ids) else ''
            f.write(f'{train},{val},{test}\n')

    print(f"   → 保存到: {output_file}")

# 保存汇总信息
summary = []
for fold in range(5):
    output_file = f'$OUTPUT_DIR/nested_splits_{fold}.csv'
    if os.path.exists(output_file):
        # 只读取有数据的行
        df_split = pd.read_csv(output_file, skip_blank_lines=True)
        # 删除空行
        df_split = df_split.dropna(how='all')
        summary.append({
            'fold': fold,
            'train': len(df_split['train'].dropna()),
            'val': len(df_split['val'].dropna()),
            'test': len(df_split['test'].dropna())
        })

if summary:
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'$OUTPUT_DIR/summary.csv', index=False)

    print("\n" + "="*50)
    print("✅ 嵌套CV划分创建完成!")
    print("="*50)
    print(f"\n📁 输出目录: $OUTPUT_DIR")
    print(f"📄 文件列表:")
    for fold in range(5):
        print(f"   - nested_splits_{fold}.csv")
    print(f"   - summary.csv")

    print(f"\n📊 汇总信息:")
    print(summary_df.to_string(index=False))
else:
    print("\n❌ 错误: 未能生成任何划分文件")

PYTHON

echo ""
echo "=========================================="
echo "脚本执行完成!"
echo "=========================================="
