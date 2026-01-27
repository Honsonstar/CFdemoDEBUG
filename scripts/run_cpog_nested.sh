#!/bin/bash
STUDY=$1
FOLD=$2
SPLIT_BASE=$3

# 构造文件路径
SPLIT_FILE="${SPLIT_BASE}/splits_${FOLD}.csv"

echo "   [Run] Python CPCG for $STUDY Fold $FOLD"
echo "   [Src] $SPLIT_FILE"

# 调用 Python 脚本
python3 run_cpog_nested_cv.py --study "$STUDY" --fold "$FOLD" --split_file "$SPLIT_FILE"

# 捕获 Python 退出代码
RET=$?
if [ $RET -ne 0 ]; then
    echo "   ❌ Fold $FOLD failed with exit code $RET"
    exit $RET
fi
