#!/bin/bash
# 同时运行 mRMR 和 Stage2 两种模式的基因比对并生成对比报告

STUDY=$1

if [ -z "$STUDY" ]; then
    echo "=========================================="
    echo "mRMR vs Stage2 基因比对对比工具"
    echo "=========================================="
    echo ""
    echo "用法: bash compare_both_modes.sh <study>"
    echo ""
    echo "示例:"
    echo "  bash compare_both_modes.sh brca"
    echo ""
    echo "功能:"
    echo "  - 运行 mRMR 模式基因比对"
    echo "  - 运行 Stage2 模式基因比对"
    echo "  - 生成两种模式的对比报告"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "mRMR vs Stage2 基因比对对比"
echo "=========================================="
echo "   癌种: $STUDY"
echo "=========================================="

# 检查必要文件
echo "\n🔍 检查必要文件..."

MISSING=0

# 检查 mRMR 文件
if [ ! -d "features/mrmr_${STUDY}" ]; then
    echo "❌ 缺少 mRMR 目录: features/mrmr_${STUDY}"
    MISSING=1
else
    echo "✅ mRMR 目录存在"
fi

# 检查 Stage2 文件
if [ ! -d "features/mrmr_stage2_${STUDY}" ]; then
    echo "❌ 缺少 Stage2 目录: features/mrmr_stage2_${STUDY}"
    MISSING=1
else
    echo "✅ Stage2 目录存在"
fi

if [ $MISSING -eq 1 ]; then
    echo "\n⚠️  缺少必要文件!"
    echo "请先运行完整的特征选择流程:"
    echo "  1. python preprocessing/CPCG_algo/stage0/run_mrmr.py --study ${STUDY} --fold all ..."
    echo "  2. bash scripts/quick_stage2_refine.sh ${STUDY}"
    exit 1
fi

echo "\n✅ 所有必要文件检查通过"

# 运行 mRMR 模式比对
echo "\n" 
echo "=========================================="
echo "📊 第1步: 运行 mRMR 模式比对"
echo "=========================================="

bash scripts/quick_mrmr_compare.sh $STUDY

MRMR_EXIT=$?

if [ $MRMR_EXIT -ne 0 ]; then
    echo "❌ mRMR 模式比对失败"
    exit 1
fi

# 运行 Stage2 模式比对
echo "\n"
echo "=========================================="
echo "📊 第2步: 运行 Stage2 模式比对"
echo "=========================================="

bash scripts/quick_mrmr_compare.sh $STUDY stage2

STAGE2_EXIT=$?

if [ $STAGE2_EXIT -ne 0 ]; then
    echo "❌ Stage2 模式比对失败"
    exit 1
fi

# 生成对比报告
echo "\n"
echo "=========================================="
echo "📊 第3步: 生成对比报告"
echo "=========================================="

# 提取关键指标
echo "\n📈 对比分析:"
echo "----------------------------------------"

# 统计基因数量
MRMR_GENES=$(tail -n +2 features/mrmr_${STUDY}/fold_0_genes.csv | wc -l)
STAGE2_GENES=$(tail -n +2 features/mrmr_stage2_${STUDY}/fold_0_genes.csv | wc -l)

echo "基因数量 (Fold 0):"
echo "  mRMR:   $MRMR_GENES 个基因"
echo "  Stage2: $STAGE2_GENES 个基因"
echo "  减少:   $((MRMR_GENES - STAGE2_GENES)) 个基因 ($(echo "scale=1; ($MRMR_GENES - $STAGE2_GENES) * 100 / $MRMR_GENES" | bc)%)"

echo ""
echo "平均重合率:"

# 提取平均重合率（需要 Python）
python3 << 'EOF'
import pandas as pd
import sys

try:
    mrmr_df = pd.read_csv('results/${STUDY}_mrmr_overlap_stats.csv')
    mrmr_avg = mrmr_df['Overlap_Rate'].mean()
    
    stage2_df = pd.read_csv('results/${STUDY}_stage2_overlap_stats.csv')
    stage2_avg = stage2_df['Overlap_Rate'].mean()
    
    print(f"  mRMR:   {mrmr_avg:.4f} ({mrmr_avg*100:.2f}%)")
    print(f"  Stage2: {stage2_avg:.4f} ({stage2_avg*100:.2f}%)")
    
    improvement = (stage2_avg - mrmr_avg) / mrmr_avg * 100
    if improvement > 0:
        print(f"  提升:   +{improvement:.1f}%")
    else:
        print(f"  变化:   {improvement:.1f}%")
except Exception as e:
    print(f"  无法计算平均重合率: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "✅ 对比分析完成!"
echo "=========================================="
echo ""
echo "📁 查看详细结果:"
echo ""
echo "mRMR 模式:"
echo "  - results/${STUDY}_mrmr_overlap_stats.csv"
echo "  - results/${STUDY}_mrmr_all_genes.csv"
echo "  - results/mrmr_gene_overlap_heatmap_${STUDY}.png (橙色)"
echo ""
echo "Stage2 模式:"
echo "  - results/${STUDY}_stage2_overlap_stats.csv"
echo "  - results/${STUDY}_stage2_all_genes.csv"
echo "  - results/stage2_gene_overlap_heatmap_${STUDY}.png (紫色)"
echo ""
echo "=========================================="
