#!/bin/bash
STUDY=$1
FOLD=$2

# ============================================================
# 【关键修复】强制使用规范化后的嵌套CV划分文件
# ============================================================
# 划分文件路径: splits/nested_cv/{cancer}/nested_splits_{fold}.csv
SPLIT_FILE="splits/nested_cv/${STUDY}/nested_splits_${FOLD}.csv"

echo "=========================================="
echo "🚀 CPCG特征筛选: ${STUDY} - Fold ${FOLD}"
echo "=========================================="
echo "   [Path] ${SPLIT_FILE}"

# 检查文件是否存在
if [ ! -f "$SPLIT_FILE" ]; then
    echo "❌ 错误: 划分文件不存在"
    echo "   预期路径: ${SPLIT_FILE}"
    echo "   请先运行: bash create_nested_splits.sh ${STUDY}"
    exit 1
fi

# 调用Python脚本 (使用完整路径)
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy
python3 preprocessing/CPCG_algo/nested_cv_wrapper.py \
    --study "$STUDY" \
    --fold "$FOLD" \
    --split_file "$SPLIT_FILE" \
    --data_root_dir "/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/datasets_csv/raw_rna_data/combine" \
    --threshold 100

# 捕获Python退出代码
RET=$?
if [ $RET -ne 0 ]; then
    echo "❌ Fold ${FOLD} 失败 (exit code: ${RET})"
    exit $RET
fi

echo "✅ Fold ${FOLD} 完成"
