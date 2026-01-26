#!/bin/bash
# 快速基因签名比对脚本

STUDY=$1

if [ -z "$STUDY" ]; then
    echo "=========================================="
    echo "快速基因签名比对"
    echo "=========================================="
    echo ""
    echo "用法: bash quick_gene_compare.sh <study>"
    echo ""
    echo "示例:"
    echo "  bash quick_gene_compare.sh blca"
    echo ""
    echo "功能:"
    echo "  - 对比全局CPCG vs 嵌套CV各折的基因重合度"
    echo "  - 对比全局CPCG vs 外部签名的基因重合度"
    echo "  - 生成详细比对报告"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "基因签名快速比对"
echo "=========================================="
echo "   癌种: $STUDY"
echo "=========================================="

# 检查必要文件
echo "\n🔍 检查必要文件..."

MISSING=0

# 检查全局CPCG结果
GLOBAL_FILE="preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_${STUDY}/tcga_${STUDY}_M2M3base_0916.csv"
if [ ! -f "$GLOBAL_FILE" ]; then
    echo "❌ 缺少: $GLOBAL_FILE"
    MISSING=1
fi

# 检查嵌套CV结果
for fold in {0..4}; do
    NESTED_FILE="features/${STUDY}/fold_${fold}_genes.csv"
    if [ ! -f "$NESTED_FILE" ]; then
        echo "❌ 缺少: $NESTED_FILE"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "\n⚠️  缺少必要文件!"
    echo "请先运行:"
    echo "  bash run_all_cpog.sh $STUDY"
    exit 1
fi

echo "✅ 所有必要文件检查通过"

# 运行比对
echo "\n🧬 开始比对基因签名..."
echo "=========================================="

python3 scripts/compare_gene_signatures.py --study $STUDY

echo ""
echo "=========================================="
echo "✅ 基因签名比对完成!"
echo "=========================================="
echo ""
echo "📁 查看结果:"
echo "   cat results/${STUDY}_overlap_stats.csv"
echo "   cat results/${STUDY}_all_genes.csv"
echo ""
echo "📊 查看热图:"
echo "   results/gene_overlap_heatmap_${STUDY}.png"
echo "=========================================="
