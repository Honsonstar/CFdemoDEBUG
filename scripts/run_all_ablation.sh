#!/bin/bash

# ====================================================================
# 多癌症类型消融实验脚本
# 一键运行 blca, brca, coadread, hnsc, stad 的消融实验
# ====================================================================

TODAY=$(date +%Y-%m-%d)
STUDIES=("blca" "brca" "coadread" "hnsc" "stad")
LOG_DIR="log/${TODAY}/ablation_all"

echo "=============================================="
echo "🚀 开始多癌症类型消融实验"
echo "📅 日期: ${TODAY}"
echo "📁 日志目录: ${LOG_DIR}"
echo "=============================================="

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 记录总日志
MASTER_LOG="${LOG_DIR}/master_ablation.log"
echo "开始多癌症类型消融实验" > "${MASTER_LOG}"
echo "时间: $(date)" >> "${MASTER_LOG}"
echo "癌症类型: ${STUDIES[*]}" >> "${MASTER_LOG}"
echo "==============================================" >> "${MASTER_LOG}"

# 颜色输出函数
print_status() {
    local color=$1
    local msg=$2
    case $color in
        green) echo -e "\033[32m${msg}\033[0m" ;;
        blue) echo -e "\033[34m${msg}\033[0m" ;;
        yellow) echo -e "\033[33m${msg}\033[0m" ;;
        red) echo -e "\033[31m${msg}\033[0m" ;;
        *) echo "${msg}" ;;
    esac
}

# 循环运行每个癌症类型的消融实验
TOTAL=${#STUDIES[@]}
COMPLETED=0
FAILED=()

for STUDY in "${STUDIES[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    echo ""
    print_status "blue" "=============================================="
    print_status "blue" "🔬 进度: [${COMPLETED}/${TOTAL}] 开始处理 ${STUDY^^}"
    print_status "blue" "=============================================="

    STUDY_LOG="${LOG_DIR}/${STUDY}_ablation.log"
    START_TIME=$(date +%s)

    # 运行消融实验
    print_status "yellow" "📝 日志文件: ${STUDY_LOG}"
    bash scripts/run_ablation_study.sh "${STUDY}" > "${STUDY_LOG}" 2>&1

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))

    # 检查结果
    if [ $? -eq 0 ]; then
        print_status "green" "✅ ${STUDY^^} 消融实验完成 (用时: ${MINUTES}分${SECONDS}秒)"
        echo "[$(date)] ${STUDY}: ✅ 完成 (${MINUTES}分${SECONDS}秒)" >> "${MASTER_LOG}"
    else
        print_status "red" "❌ ${STUDY^^} 消融实验失败"
        echo "[$(date)] ${STUDY}: ❌ 失败" >> "${MASTER_LOG}"
        FAILED+=("${STUDY}")
    fi

    # 显示摘要
    if [ -f "results/ablation/${STUDY}/final_comparison.csv" ]; then
        echo ""
        print_status "blue" "📊 ${STUDY^^} 结果摘要:"
        tail -20 "${STUDY_LOG}" | grep -A 10 "平均 C-Index" | head -15
    fi

    echo "" >> "${MASTER_LOG}"
done

# 最终汇总
echo ""
print_status "blue" "=============================================="
print_status "blue" "🎉 所有消融实验完成!"
print_status "blue" "=============================================="
echo ""
echo "📊 总共: ${TOTAL} 个癌症类型"
echo "✅ 成功: $((TOTAL - ${#FAILED[@]})) 个"
echo "❌ 失败: ${#FAILED[@]} 个"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "失败的癌症类型: ${FAILED[*]}"
fi

echo ""
print_status "blue" "📁 结果汇总:"
echo "   - 总日志: ${MASTER_LOG}"
for STUDY in "${STUDIES[@]}"; do
    echo "   - ${STUDY^^}: log/${TODAY}/${STUDY}_ablation.log"
done

echo ""
print_status "blue" "📊 对比表格:"
for STUDY in "${STUDIES[@]}"; do
    if [ -f "results/ablation/${STUDY}/final_comparison.csv" ]; then
        echo "   ✓ ${STUDY^^}: results/ablation/${STUDY}/final_comparison.csv"
    fi
done

echo ""
print_status "green" "🎯 运行完成! 查看详细日志请使用:"
echo "   cat ${MASTER_LOG}"
for STUDY in "${STUDIES[@]}"; do
    echo "   cat log/${TODAY}/${STUDY}_ablation.log"
done

echo ""
print_status "yellow" "💡 提示: 可以使用以下命令查看所有结果的C-Index对比:"
echo "   python3 scripts/compare_ablation_results.py"
