#!/bin/bash
STUDY=$1

if [ -z "$STUDY" ]; then
    echo "Usage: bash scripts/run_all_cpog.sh <study>"
    exit 1
fi

# 硬编码正确的外部数据路径
SPLIT_BASE="/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_${STUDY}"

echo "=========================================="
echo "🚀 启动筛选: $STUDY"
echo "📂 数据源: $SPLIT_BASE"
echo "=========================================="

# 检查数据是否存在
if [ ! -f "${SPLIT_BASE}/splits_0.csv" ]; then
    echo "❌ 错误: 找不到划分文件 ${SPLIT_BASE}/splits_0.csv"
    exit 1
fi

mkdir -p "features/${STUDY}"

# 循环运行 5 折
for fold in {0..4}; do
    echo ""
    echo ">>> Fold $fold <<<"
    
    # 【关键修复】使用 scripts/ 前缀调用子脚本
    bash scripts/run_cpog_nested.sh "$STUDY" "$fold" "$SPLIT_BASE"
    
    if [ $? -ne 0 ]; then
        echo "❌ Fold $fold 失败！停止当前癌种任务。"
        exit 1
    fi
done

echo ""
echo "✅ $STUDY 所有折筛选完成。"
