#!/bin/bash
STUDY=$1

if [ -z "$STUDY" ]; then
    echo "Usage: bash scripts/run_all_cpog.sh <study>"
    exit 1
fi

# 外部数据绝对路径
SPLIT_BASE="/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_${STUDY}"

echo "=========================================="
echo "🚀 启动筛选任务: $STUDY"
echo "📂 读取外部划分: $SPLIT_BASE"
echo "=========================================="

if [ ! -d "$SPLIT_BASE" ]; then
    echo "❌ 错误: 目录不存在 $SPLIT_BASE"
    exit 1
fi

# 循环 5 折
for fold in {0..4}; do
    echo ""
    echo ">>> Processing Fold $fold..."
    
    # 调用子脚本 (确保使用 scripts/ 前缀)
    bash scripts/run_cpog_nested.sh "$STUDY" "$fold" "$SPLIT_BASE"
    
    # 检查退出代码
    if [ $? -ne 0 ]; then
        echo "❌ 严重错误: Fold $fold 失败。停止任务以避免产生错误数据。"
        exit 1
    fi
done

echo ""
echo "✅✅✅ $STUDY 所有折筛选完毕！"
