#!/bin/bash

# ====================================================================
# 批量运行 CPCG 筛选脚本
# 一次性运行 hnsc, coadread, stad 三种癌症类型的5折筛选
# ====================================================================

STUDIES=("hnsc" "coadread" "stad")
LOG_DIR="log/$(date +%Y-%m-%d)/cpog_all"

echo "=============================================="
echo "🚀 开始批量 CPCG 筛选"
echo "📅 日期: $(date +%Y-%m-%d)"
echo "📁 日志目录: ${LOG_DIR}"
echo "=============================================="

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master_cpog.log"

for STUDY in "${STUDIES[@]}"; do
    echo ""
    echo "=============================================="
    echo "🔬 开始处理 ${STUDY^^}"
    echo "=============================================="
    STUDY_LOG="${LOG_DIR}/${STUDY}_cpog.log"

    START_TIME=$(date +%s)
    bash scripts/run_all_cpog.sh "${STudy}" > "${STUDY_LOG}" 2>&1
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))

    if [ $? -eq 0 ]; then
        echo "✅ ${STUDY^^} 完成 (用时: ${MINUTES}分${SECONDS}秒)"
    else
        echo "❌ ${STUDY^^} 失败"
    fi

    echo "[$(date)] ${STUDY}: 完成 (${MINUTES}分${SECONDS}秒)" >> "${MASTER_LOG}"
done

echo ""
echo "=============================================="
echo "🎉 所有 CPCG 筛选完成!"
echo "=============================================="
echo ""
echo "📁 日志文件:"
for STUDY in "${STUDIES[@]}"; do
    echo "   - ${STUDY^^}: log/$(date +%Y-%m-%d)/cpog_all/${STUDY}_cpog.log"
done
