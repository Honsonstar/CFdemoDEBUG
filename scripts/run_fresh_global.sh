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

# 2. 运行 CPCG (使用新的 wrapper)
echo "2️⃣  运行 CPCG 筛选 (这可能需要几分钟)..."
# 使用 fold 999 作为标记
OUTPUT_FILE=$(python3 preprocessing/CPCG_algo/nested_cv_wrapper.py \
    --study "$STUDY" \
    --fold 999 \
    --split_file "$SPLIT_FILE" \
    --data_root_dir "datasets_csv/raw_rna_data/combine" 2>&1 | tail -1)

# 提取输出文件路径（最后一行的 "输出文件: /tmp/xxx.csv"）
SRC_FILE=$(echo "$OUTPUT_FILE" | grep "输出文件:" | awk '{print $NF}')

if [ -z "$SRC_FILE" ] || [ ! -f "$SRC_FILE" ]; then
    echo "❌ 筛选失败，未找到结果文件"
    echo "原始输出: $OUTPUT_FILE"
    exit 1
fi

# 复制到 features 目录
mkdir -p "features/${STUDY}"
DEST_FILE="features/${STUDY}/fold_999_genes.csv"
cp "$SRC_FILE" "$DEST_FILE"
echo "✅ 已复制结果到: $DEST_FILE"

# 3. 归档结果
echo "3️⃣  归档结果..."
DEST_DIR="results/comparison/${STUDY}"
mkdir -p "$DEST_DIR"
FINAL_FILE="${DEST_DIR}/global_genes.csv"

if [ -f "$DEST_FILE" ]; then
    cp "$DEST_FILE" "$FINAL_FILE"
    echo "✅ 新的全局基准已保存: $FINAL_FILE"

    # 清理临时文件
    rm "$DEST_FILE"
    rm -rf "$SPLIT_DIR"
else
    echo "❌ 筛选失败，未找到结果文件"
    exit 1
fi

# 4. 重新运行对比
echo "4️⃣  重新运行对比..."
bash scripts/quick_gene_compare.sh "$STUDY"
