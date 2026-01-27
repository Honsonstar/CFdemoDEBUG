#!/bin/bash
# 优化后的CPCG特征筛选脚本

STUDY=$1
FOLD=$2

if [ -z "$STUDY" ] || [ -z "$FOLD" ]; then
    echo "=========================================="
    echo "用法: bash run_cpog_optimized.sh <study> <fold>"
    echo ""
    echo "示例:"
    echo "  bash run_cpog_optimized.sh blca 0"
    echo "  bash run_cpog_optimized.sh brca 3"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "运行优化后的CPCG特征筛选 (嵌套CV)"
echo "=========================================="
echo "   癌种: $STUDY"
echo "   折数: $FOLD"
echo "   优化特性:"
echo "     - 并行化: joblib多核心处理"
echo "     - 阈值: threshold=100"
echo "     - 进度条: tqdm实时显示"
echo "     - 兜底机制: 零特征自动回退"
echo "=========================================="

# 检查嵌套划分文件
SPLITS_FILE="splits/nested_cv/${STUDY}/nested_splits_${FOLD}.csv"
if [ ! -f "$SPLITS_FILE" ]; then
    echo "❌ 错误: 找不到划分文件 $SPLITS_FILE"
    echo "请先运行: bash create_nested_splits.sh $STUDY"
    exit 1
fi

# 创建特征输出目录
FEATURES_DIR="features/${STUDY}"
mkdir -p "$FEATURES_DIR"

# 运行Python脚本执行CPCG筛选（使用优化后的算法）
python3 << PYTHON
import sys
sys.path.insert(0, 'preprocessing/CPCG_algo')

from nested_cv_wrapper import NestedCVFeatureSelector
import pandas as pd
import time

print("\n🚀 启动优化后的CPCG特征筛选...")

# 读取划分
splits_df = pd.read_csv('$SPLITS_FILE')
train_ids = splits_df['train'].dropna().tolist()
val_ids = splits_df['val'].dropna().tolist()
test_ids = splits_df['test'].dropna().tolist()

print(f"\n📊 数据统计:")
print(f"   训练集: {len(train_ids)} 样本")
print(f"   验证集: {len(val_ids)} 样本")
print(f"   测试集: {len(test_ids)} 样本")

# 创建特征选择器（使用优化后的参数）
selector = NestedCVFeatureSelector(
    study='$STUDY',
    data_root_dir='preprocessing/CPCG_algo/raw_data',
    threshold=100,
    n_jobs=-1
)

print(f"\n⚙️  优化配置:")
print(f"   并行作业数: -1 (使用所有CPU核心)")
print(f"   基因筛选阈值: 100")
print(f"   进度条: 启用")
print(f"   兜底机制: 启用")

start_time = time.time()

try:
    with selector:
        feature_file = selector.select_features_for_fold(
            fold=$FOLD,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids
        )

    elapsed = time.time() - start_time

    print(f"\n✅ CPCG特征筛选完成!")
    print(f"   输出文件: {feature_file}")
    print(f"   总耗时: {elapsed:.2f} 秒")

    # 验证文件
    import os
    if os.path.exists(feature_file):
        df = pd.read_csv(feature_file)
        print(f"   基因数量: {df.shape[1] - 1}")
        print(f"   样本数量: {df.shape[0]}")
    else:
        print(f"   ⚠️  文件不存在")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n❌ 运行出错 (耗时: {elapsed:.2f}s)")
    print(f"   错误: {str(e)[:200]}")
    import traceback
    traceback.print_exc()

PYTHON

echo ""
echo "=========================================="
echo "CPCG筛选完成!"
echo "=========================================="
