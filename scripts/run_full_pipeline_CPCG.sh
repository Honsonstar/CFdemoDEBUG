#!/bin/bash

# ====================================================================
# 完整流程批处理脚本：CPCG筛选 → 消融实验
# ====================================================================
#
# 使用方法:
#   bash scripts/run_full_pipeline_CPCG.sh           # 运行所有癌种
#   bash scripts/run_full_pipeline_CPCG.sh brca     # 运行单个癌种
#
# 流程:
#   1. CPCG 特征筛选 (run_all_cpog.sh)
#   2. 消融实验 (run_ablation_study_cpcg.sh)
#
# 特征路径: features/{study}/fold_{0-4}_genes.csv  # CPCG筛选后的特征
# 划分路径: splits/nested_cv/{study}/               # 嵌套CV划分
# 结果路径: results/ablation_cpcg/{study}/          # 消融实验结果 (CPCG)
# 日志路径: log/batch_pipeline_cpcg_{date}/        # 批处理日志
#
# 说明:
#   - 使用 nested_cv 划分，只用训练集样本筛选特征，无数据泄露
#   - CPCG特征使用独立的消融实验脚本 run_ablation_study_cpcg.sh
# ====================================================================

DEFAULT_STUDIES="brca blca coadread stad hnsc"
STUDIES=${1:-$DEFAULT_STUDIES}

echo "=============================================="
echo "🚀 批量运行完整流程 (CPCG特征)"
echo "=============================================="
echo "📋 癌症类型: $STUDIES"

SPLIT_DIR="splits/nested_cv"
TODAY=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="log/batch_pipeline_cpcg_${TODAY}"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/main.log"

echo "📁 日志目录: ${LOG_DIR}" | tee -a "${MAIN_LOG}"

log_step() {
    echo "[$(date '+%H:%M:%S')] [$1] $2 - $3" | tee -a "${MAIN_LOG}"
}

check_success() { [ $? -eq 0 ]; }

TOTAL_START=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_STUDIES=""

for study in $STUDIES; do
    STUDY_START=$(date +%s)
    echo "=============================================="
    echo "🧬 开始处理: ${study^^}"
    echo "=============================================="
    STUDY_LOG="${LOG_DIR}/${study}_full.log"

    log_step "$study" "检查数据划分" "开始"
    if [ ! -d "${SPLIT_DIR}/${study}" ]; then
        echo "❌ 错误: 找不到 ${SPLIT_DIR}/${study}" | tee -a "${STUDY_LOG}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_STUDIES="${FAILED_STUDIES} ${study}"
        continue
    fi
    log_step "$study" "检查数据划分" "通过"

    # CPCG特征筛选
    log_step "$study" "CPCG特征筛选" "开始"
    CPCG_LOG="${LOG_DIR}/${study}_cpcg.log"
    bash scripts/run_all_cpog.sh "${study}" > "${CPCG_LOG}" 2>&1
    if check_success; then
        log_step "$study" "CPCG特征筛选" "完成"
        ls -lh "features/${study}"/fold_*_genes.csv 2>/dev/null | awk '{print "      " $9}' | tee -a "${STUDY_LOG}"
    else
        log_step "$study" "CPCG特征筛选" "失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_STUDIES="${FAILED_STUDIES} ${study}"
        continue
    fi

    # 消融实验 (使用CPCG特征)
    log_step "$study" "消融实验(CPCG)" "开始"
    ABLATION_LOG="${LOG_DIR}/${study}_ablation.log"
    bash scripts/run_ablation_study_cpcg.sh "${study}" > "${ABLATION_LOG}" 2>&1
    if check_success; then
        log_step "$study" "消融实验(CPCG)" "完成"
        RESULT_CSV="results/ablation_cpcg/${study}/final_comparison.csv"
        if [ -f "$RESULT_CSV" ]; then
            python3 << EOF | tee -a "${STUDY_LOG}"
import pandas as pd
df = pd.read_csv("${RESULT_CSV}")
print(f"      Gene: {df['Gene_C_Index'].mean():.4f}")
print(f"      Text: {df['Text_C_Index'].mean():.4f}")
print(f"      Fusion: {df['Fusion_C_Index'].mean():.4f}")
EOF
        fi
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log_step "$study" "消融实验" "失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_STUDIES="${FAILED_STUDIES} ${study}"
    fi

    STUDY_END=$(date +%s)
    echo "   ⏱️  耗时: $((($STUDY_END - $STUDY_START) / 60))分$((($STUDY_END - $STUDY_START) % 60))秒" | tee -a "${STUDY_LOG}"
done

TOTAL_END=$(date +%s)
echo ""
echo "=============================================="
echo "🎉 批处理完成!"
echo "   ✅ 成功: ${SUCCESS_COUNT} 个"
echo "   ❌ 失败: ${FAIL_COUNT} 个"
echo "   ⏱️  总耗时: $((($TOTAL_END - $TOTAL_START) / 60))分$((($TOTAL_END - $TOTAL_START) % 60))秒"
echo "   📁 日志: ${LOG_DIR}"
echo "=============================================="

[ ${FAIL_COUNT} -eq 0 ] && exit 0 || exit 1
