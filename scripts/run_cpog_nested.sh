#!/bin/bash
STUDY=$1
FOLD=$2
SPLIT_DIR=$3

echo "   [Sub-Task] Running CPCG for $STUDY Fold $FOLD..."
echo "   Split Dir: $SPLIT_DIR"

# 关键修正：确保参数之间有空格，且引用正确
# 同时也打印出即将执行的命令以便调试
CMD="python3 run_real_nested_cv.py --study ${STUDY} --fold ${FOLD} --split_dir ${SPLIT_DIR}"
echo "   Executing: $CMD"

$CMD

if [ $? -ne 0 ]; then
    echo "   ❌ Python script failed for Fold $FOLD"
    exit 1
fi
