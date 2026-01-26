#!/bin/bash
# 一键运行所有折的CPCG筛选

STUDY=$1

if [ -z "$STUDY" ]; then
    echo "=========================================="
    echo "一键运行所有折的CPCG筛选"
    echo "=========================================="
    echo ""
    echo "用法: bash run_all_cpog.sh <study>"
    echo ""
    echo "示例:"
    echo "  bash run_all_cpog.sh blca"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "一键运行所有折的CPCG筛选"
echo "=========================================="
echo "   癌种: $STUDY"
echo "=========================================="

# 检查嵌套划分文件
echo "\n🔍 检查嵌套划分文件..."
MISSING=0

for fold in {0..4}; do
    if [ ! -f "splits/nested_cv/${STUDY}/nested_splits_${fold}.csv" ]; then
        echo "❌ 缺少: splits/nested_cv/${STUDY}/nested_splits_${fold}.csv"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "\n⚠️  缺少划分文件，请先运行:"
    echo "   bash create_nested_splits.sh $STUDY"
    exit 1
fi

echo "✅ 划分文件检查通过"

# 创建特征目录
FEATURES_DIR="features/${STUDY}"
mkdir -p "$FEATURES_DIR"

echo "\n📁 特征输出目录: $FEATURES_DIR"

# 并行或串行运行所有折
echo "\n🚀 开始运行所有折的CPCG筛选..."
echo "=========================================="

# 使用串行执行 (避免资源竞争)
START_TIME=$(date +%s)

for fold in {0..4}; do
    echo ""
    echo ">>> 折数: $fold / 4 <<<"
    echo "------------------------------------------"
    
    FOLD_START=$(date +%s)
    
    # 运行CPCG筛选
    bash run_cpog_nested.sh $STUDY $fold
    
    FOLD_END=$(date +%s)
    FOLD_DURATION=$((FOLD_END - FOLD_START))
    
    echo ""
    echo "✅ 折 $fold 完成 (耗时: ${FOLD_DURATION}s)"
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "✅ 所有折CPCG筛选完成!"
echo "=========================================="
echo "   总耗时: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m)"
echo ""

# 验证结果
echo "🔍 验证结果..."
ALL_PRESENT=1

for fold in {0..4}; do
    GENE_FILE="${FEATURES_DIR}/fold_${fold}_genes.csv"
    if [ -f "$GENE_FILE" ]; then
        GENE_COUNT=$(wc -l < "$GENE_FILE")
        echo "   ✓ Fold $fold: $GENE_COUNT 个基因"
    else
        echo "   ❌ Fold $fold: 基因文件缺失"
        ALL_PRESENT=0
    fi
done

if [ $ALL_PRESENT -eq 1 ]; then
    echo "\n✅ 所有基因文件验证通过!"
else
    echo "\n❌ 部分基因文件缺失，请检查错误!"
    exit 1
fi

# 生成汇总
python3 << PYTHON
import pandas as pd
import os

study = '$STUDY'
features_dir = 'features/${STUDY}'

print("\n📊 生成基因筛选汇总...")

gene_counts = []
total_unique_genes = set()

for fold in range(5):
    gene_file = f'{features_dir}/fold_{fold}_genes.csv'
    if os.path.exists(gene_file):
        df = pd.read_csv(gene_file)
        gene_count = len(df)
        gene_counts.append(gene_count)
        total_unique_genes.update(df['gene'].tolist())
        print(f"   Fold {fold}: {gene_count} 个基因")
    else:
        print(f"   Fold {fold}: 文件不存在")
        gene_counts.append(0)

print(f"\n📈 统计:")
print(f"   平均每折基因数: {sum(gene_counts)/len(gene_counts):.1f}")
print(f"   总唯一基因数: {len(total_unique_genes)}")

# 计算折间重叠
print(f"\n🔍 折间基因重叠分析:")
overlaps = []
for i in range(5):
    for j in range(i+1, 5):
        df_i = pd.read_csv(f'{features_dir}/fold_{i}_genes.csv')
        df_j = pd.read_csv(f'{features_dir}/fold_{j}_genes.csv')
        genes_i = set(df_i['gene'])
        genes_j = set(df_j['gene'])
        overlap = len(genes_i & genes_j)
        overlap_pct = overlap / len(genes_i | genes_j) * 100
        overlaps.append(overlap_pct)
        print(f"   Fold {i} vs {j}: {overlap_pct:.1f}% 重叠")

avg_overlap = sum(overlaps) / len(overlaps)
print(f"\n   平均重叠率: {avg_overlap:.1f}%")

if avg_overlap < 50:
    print("   ⚠️  重叠率较低，说明每折筛选差异较大")
elif avg_overlap > 80:
    print("   ⚠️  重叠率较高，说明特征较稳定但可能存在泄露风险")
else:
    print("   ✅ 重叠率适中，特征有一定稳定性")

# 保存汇总
summary = {
    'fold': list(range(5)),
    'gene_count': gene_counts,
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv(f'{features_dir}/summary.csv', index=False)

print(f"\n✅ 汇总保存到: {features_dir}/summary.csv")

PYTHON

echo ""
echo "=========================================="
echo "🎉 CPCG筛选流程全部完成!"
echo "=========================================="
echo ""
echo "📁 输出目录: $FEATURES_DIR"
echo "   - fold_0_genes.csv 到 fold_4_genes.csv"
echo "   - summary.csv (汇总)"
echo ""
echo "✅ 接下来运行:"
echo "   bash train_all_folds.sh $STUDY"
echo ""
echo "或运行完整对比实验:"
echo "   bash compare_global_vs_nested.sh $STUDY"
echo "=========================================="
