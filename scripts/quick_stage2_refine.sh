#!/bin/bash
# Stage 2 特征精炼快速运行脚本

STUDY=$1

if [ -z "$STUDY" ]; then
    echo "=========================================="
    echo "Stage 2 特征精炼（PC算法）"
    echo "=========================================="
    echo ""
    echo "用法: bash quick_stage2_refine.sh <study>"
    echo ""
    echo "示例:"
    echo "  bash quick_stage2_refine.sh blca"
    echo "  bash quick_stage2_refine.sh brca"
    echo ""
    echo "功能:"
    echo "  - 对 mRMR 筛选的基因进行 Stage 2 (PC算法) 二次筛选"
    echo "  - 提取与生存时间 (OS) 直接相关的基因 (Markov Blanket)"
    echo "  - 生成精炼后的特征文件"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "Stage 2 特征精炼 (PC算法)"
echo "=========================================="
echo "   癌种: $STUDY"
echo "=========================================="

# 检查必要文件
echo "\n🔍 检查必要文件..."

MISSING=0

# 检查 mRMR 输入目录
MRMR_DIR="features/mrmr_${STUDY}"
if [ ! -d "$MRMR_DIR" ]; then
    echo "❌ 缺少 mRMR 输入目录: $MRMR_DIR"
    MISSING=1
else
    echo "✅ mRMR 输入目录: $MRMR_DIR"
    
    # 检查各折 mRMR 基因文件
    for fold in {0..4}; do
        MRMR_FILE="${MRMR_DIR}/fold_${fold}_genes.csv"
        if [ ! -f "$MRMR_FILE" ]; then
            echo "❌ 缺少: $MRMR_FILE"
            MISSING=1
        else
            echo "  ✓ Fold ${fold}: $MRMR_FILE"
        fi
    done
fi

# 检查临床数据
CLINICAL_DIR="datasets_csv/clinical_data"
CLINICAL_FILE="${CLINICAL_DIR}/tcga_${STUDY}_clinical.csv"
if [ ! -f "$CLINICAL_FILE" ]; then
    # 尝试另一种路径格式
    CLINICAL_FILE="${CLINICAL_DIR}/tcga_${STUDY}/clinical.CSV"
    if [ ! -f "$CLINICAL_FILE" ]; then
        echo "❌ 缺少临床数据文件: ${CLINICAL_DIR}/tcga_${STUDY}_clinical.csv"
        MISSING=1
    else
        echo "✅ 临床数据: $CLINICAL_FILE"
    fi
else
    echo "✅ 临床数据: $CLINICAL_FILE"
fi

if [ $MISSING -eq 1 ]; then
    echo "\n⚠️  缺少必要文件!"
    echo "请先运行 mRMR 特征选择:"
    echo "  python preprocessing/CPCG_algo/stage0/run_mrmr.py --study ${STUDY} --fold all ..."
    exit 1
fi

echo "\n✅ 所有必要文件检查通过"

# 运行 Stage 2 精炼
echo "\n🧬 开始 Stage 2 特征精炼（PC算法）..."
echo "=========================================="

python3 preprocessing/CPCG_algo/stage0/run_stage2_refinement.py \
    --study $STUDY \
    --fold all \
    --clinical_dir datasets_csv/clinical_data

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Stage 2 特征精炼完成!"
    echo "=========================================="
    echo ""
    echo "📁 查看结果:"
    echo "   ls -lh features/mrmr_stage2_${STUDY}/"
    echo ""
    echo "📊 对比 mRMR vs Stage2 基因数量:"
    echo "   # mRMR 基因数"
    echo "   head -n 1 features/mrmr_${STUDY}/fold_0_genes.csv | awk -F',' '{print NF-1}'"
    echo ""
    echo "   # Stage2 精炼后基因数"
    echo "   head -n 1 features/mrmr_stage2_${STUDY}/fold_0_genes.csv | awk -F',' '{print NF-1}'"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Stage 2 特征精炼失败!"
    echo "=========================================="
    exit 1
fi
