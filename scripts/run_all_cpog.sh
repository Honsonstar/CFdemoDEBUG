#!/bin/bash
STUDY=$1

if [ -z "$STUDY" ]; then
    echo "Usage: bash scripts/run_all_cpog.sh <study>"
    echo "示例: bash scripts/run_all_cpog.sh stad"
    exit 1
fi

echo "=========================================="
echo "🚀 启动CPCG筛选任务: $STUDY"
echo "   划分文件: splits/nested_cv/${STUDY}/nested_splits_{fold}.csv"
echo "=========================================="

# 检查划分目录是否存在
SPLIT_DIR="splits/nested_cv/${STUDY}"
if [ ! -d "$SPLIT_DIR" ]; then
    echo "❌ 错误: 划分目录不存在 $SPLIT_DIR"
    echo "请先运行: bash create_nested_splits.sh $STUDY"
    exit 1
fi

# 循环5折
for fold in {0..4}; do
    echo ""
    echo ">>> 处理 Fold $fold..."
    bash scripts/run_cpog_nested.sh "$STUDY" "$fold"

    # 检查退出代码
    if [ $? -ne 0 ]; then
        echo "❌ 严重错误: Fold $fold 失败"
        exit 1
    fi
done

echo ""
echo "✅✅✅ $STUDY 所有折筛选完毕！"
echo "   结果保存在: features/${STUDY}/fold_*_genes.csv"
