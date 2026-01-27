#!/bin/bash
STUDY=$1

if [ -z "$STUDY" ]; then
    echo "Usage: bash scripts/run_fresh_global.sh <study>"
    exit 1
fi

echo "=========================================="
echo "🚀 启动 Fresh Global CPCG 筛选: $STUDY"
echo "   (使用全量数据生成新的对比基准)"
echo "=========================================="

# 1. 准备全量数据的 Split 文件
echo "1️⃣  生成全量 Split 文件..."
SPLIT_DIR="temp_global_split"
mkdir -p $SPLIT_DIR
SPLIT_FILE="${SPLIT_DIR}/splits_global.csv"

python3 -c "
import pandas as pd
import os

# 读取临床数据获取所有样本ID
clinical_file = f'datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv'
if not os.path.exists(clinical_file):
    print(f'❌ 找不到临床文件: {clinical_file}')
    exit(1)

df = pd.read_csv(clinical_file)
# 截取 ID (TCGA-XX-XXXX)
ids = df['case_id'].str[:12].unique()

# 创建全量 split (所有样本都在 train)
split_df = pd.DataFrame({'train': ids})
split_df.to_csv('$SPLIT_FILE', index=False)
print(f'✅ 全量样本数: {len(ids)}')
"

if [ ! -f "$SPLIT_FILE" ]; then
    echo "❌ Split 文件生成失败"
    exit 1
fi

# 2. 运行 CPCG (借用 run_cpog_nested_cv.py)
echo "2️⃣  运行 CPCG 筛选 (这可能需要几分钟)..."
# 使用 fold 999 作为标记
python3 run_cpog_nested_cv.py --study "$STUDY" --fold 999 --split_file "$SPLIT_FILE"

# 3. 归档结果
echo "3️⃣  归档结果..."
# run_cpog_nested_cv.py 默认输出到 features/$STUDY/fold_999_genes.csv
SRC_FILE="features/${STUDY}/fold_999_genes.csv"
DEST_DIR="results/comparison/${STUDY}"
mkdir -p "$DEST_DIR"
DEST_FILE="${DEST_DIR}/global_genes.csv"

if [ -f "$SRC_FILE" ]; then
    cp "$SRC_FILE" "$DEST_FILE"
    echo "✅ 新的全局基准已保存: $DEST_FILE"
    
    # 清理临时文件
    rm "$SRC_FILE"
    rm -rf "$SPLIT_DIR"
else
    echo "❌ 筛选失败，未找到结果文件"
    exit 1
fi

# 4. 重新运行对比
echo "4️⃣  重新运行对比..."
bash scripts/quick_gene_compare.sh "$STUDY"
