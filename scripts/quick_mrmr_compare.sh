#!/bin/bash
# MRMR基因签名快速比对脚本

STUDY=$1
MODE=$2  # 可选: "stage2" 表示使用 Stage2 精炼后的基因

if [ -z "$STUDY" ]; then
    echo "=========================================="
    echo "MRMR基因签名快速比对"
    echo "=========================================="
    echo ""
    echo "用法: bash quick_mrmr_compare.sh <study> [mode]"
    echo ""
    echo "参数:"
    echo "  study    - 癌种名称 (必需)"
    echo "  mode     - 可选: 'stage2' 使用Stage2精炼后的基因"
    echo "             默认: 使用mRMR原始筛选的基因"
    echo ""
    echo "示例:"
    echo "  bash quick_mrmr_compare.sh brca          # 比对mRMR原始基因"
    echo "  bash quick_mrmr_compare.sh brca stage2   # 比对Stage2精炼基因"
    echo ""
    echo "功能:"
    echo "  - 对比全局CPCG vs 嵌套CV各折基因重合度"
    echo "  - 对比不同折间的基因一致性"
    echo "  - 生成详细比对报告和热力图"
    echo "=========================================="
    exit 1
fi

# 确定使用哪种模式
if [ "$MODE" = "stage2" ]; then
    USE_STAGE2="--stage2"
    FEATURE_DIR="mrmr_stage2_${STUDY}"
    MODE_NAME="MRMR + Stage2 (PC算法)"
else
    USE_STAGE2=""
    FEATURE_DIR="mrmr_${STUDY}"
    MODE_NAME="MRMR"
fi

echo "=========================================="
echo "基因签名快速比对"
echo "=========================================="
echo "   癌种: $STUDY"
echo "   模式: $MODE_NAME"
echo "=========================================="

# 检查必要文件
echo "\n🔍 检查必要文件..."

MISSING=0

# 检查全局CPCG结果（可选）
GLOBAL_FILE="preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_${STUDY}/tcga_${STUDY}_M2M3base_0916.csv"
if [ ! -f "$GLOBAL_FILE" ]; then
    echo "⚠️  全局文件不存在: $GLOBAL_FILE"
    echo "    (可选，不影响MRMR折间比对)"
else
    echo "✅ 全局CPCG文件: $GLOBAL_FILE"
fi

# 检查特征文件
FEATURE_PATH="features/${FEATURE_DIR}"
if [ ! -d "$FEATURE_PATH" ]; then
    echo "❌ 缺少目录: $FEATURE_PATH"
    MISSING=1
else
    echo "✅ 特征目录: $FEATURE_PATH"
    
    # 检查各折基因文件
    for fold in {0..4}; do
        GENE_FILE="${FEATURE_PATH}/fold_${fold}_genes.csv"
        if [ ! -f "$GENE_FILE" ]; then
            echo "❌ 缺少: $GENE_FILE"
            MISSING=1
        else
            echo "  ✓ Fold ${fold}: $GENE_FILE"
        fi
    done
fi

if [ $MISSING -eq 1 ]; then
    echo "\n⚠️  缺少必要文件!"
    if [ "$MODE" = "stage2" ]; then
        echo "请先运行 Stage2 特征精炼:"
        echo "  bash scripts/quick_stage2_refine.sh ${STUDY}"
    else
        echo "请先运行 MRMR 特征选择:"
        echo "  python preprocessing/CPCG_algo/stage0/run_mrmr.py --study ${STUDY} --fold all ..."
    fi
    exit 1
fi

echo "\n✅ 所有必要文件检查通过"

# 运行比对
echo "\n🧬 开始比对基因签名..."
echo "=========================================="

python3 scripts/compare_mrmr_gene_signatures.py --study $STUDY $USE_STAGE2

# 确定输出文件后缀
if [ "$MODE" = "stage2" ]; then
    SUFFIX="stage2"
else
    SUFFIX="mrmr"
fi

echo ""
echo "=========================================="
echo "✅ 基因签名比对完成!"
echo "=========================================="
echo ""
echo "📁 查看结果:"
echo "   cat results/${STUDY}_${SUFFIX}_overlap_stats.csv"
echo "   cat results/${STUDY}_${SUFFIX}_all_genes.csv"
echo ""
echo "📊 查看热图:"
echo "   results/${SUFFIX}_gene_overlap_heatmap_${STUDY}.png"
echo "=========================================="
