#!/bin/bash

# ====================================================================
# 完整流程批处理脚本：mRMR → Stage2 → 消融实验
# 支持多个癌症类型一键运行
# ====================================================================

# 默认运行的癌症类型
DEFAULT_STUDIES="brca blca luad stad hnsc"

# 使用命令行参数或默认值
STUDIES=${1:-$DEFAULT_STUDIES}

echo "=============================================="
echo "🚀 批量运行完整流程"
echo "=============================================="
echo "📋 癌症类型: $STUDIES"
echo "📊 流程步骤:"
echo "   1️⃣  mRMR 特征选择 (k=30)"
echo "   2️⃣  Stage2 PC算法精炼"
echo "   3️⃣  消融实验 (Gene/Text/Fusion)"
echo "=============================================="
echo ""

# 配置路径（从项目根目录执行）
SPLIT_DIR="splits/nested_cv"
DATA_ROOT_DIR="datasets_csv/raw_rna_data/combine"
CLINICAL_DIR="datasets_csv/clinical_data"
THRESHOLD=30

# 日志目录
TODAY=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="log/batch_pipeline_${TODAY}"
mkdir -p "${LOG_DIR}"

MAIN_LOG="${LOG_DIR}/main.log"

echo "📁 日志目录: ${LOG_DIR}" | tee -a "${MAIN_LOG}"
echo "" | tee -a "${MAIN_LOG}"

# ====================================================================
# 辅助函数
# ====================================================================

log_step() {
    local study=$1
    local step=$2
    local status=$3
    echo "[$(date '+%H:%M:%S')] [$study] $step - $status" | tee -a "${MAIN_LOG}"
}

check_success() {
    if [ $? -eq 0 ]; then
        echo "✅"
        return 0
    else
        echo "❌"
        return 1
    fi
}

# ====================================================================
# 主流程循环
# ====================================================================

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
    
    # ----------------------------------------------------------------
    # Step 1: 检查数据划分
    # ----------------------------------------------------------------
    log_step "$study" "检查数据划分" "开始"
    
    if [ ! -d "${SPLIT_DIR}/${study}" ]; then
        echo "❌ 错误: 找不到 ${SPLIT_DIR}/${study}，请先运行 create_nested_splits.sh ${study}" | tee -a "${STUDY_LOG}"
        log_step "$study" "检查数据划分" "失败 - 缺少划分文件"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_STUDIES="${FAILED_STUDIES} ${study}"
        continue
    fi
    
    log_step "$study" "检查数据划分" "通过"
    
    # ----------------------------------------------------------------
    # Step 2: mRMR 特征选择
    # ----------------------------------------------------------------
    log_step "$study" "mRMR特征选择" "开始 (k=${THRESHOLD})"
    
    MRMR_LOG="${LOG_DIR}/${study}_mrmr.log"
    
    python3 preprocessing/CPCG_algo/stage0/run_mrmr.py \
        --study "${study}" \
        --fold all \
        --split_dir "${SPLIT_DIR}" \
        --data_root_dir "${DATA_ROOT_DIR}" \
        --clinical_dir "${CLINICAL_DIR}" \
        --threshold ${THRESHOLD} \
        > "${MRMR_LOG}" 2>&1
    
    if check_success; then
        log_step "$study" "mRMR特征选择" "完成"
        
        # 显示生成的文件
        echo "   📂 生成文件:" | tee -a "${STUDY_LOG}"
        ls -lh features/mrmr_${study}/fold_*.csv 2>/dev/null | awk '{print "      " $9}' | tee -a "${STUDY_LOG}"
    else
        log_step "$study" "mRMR特征选择" "失败"
        echo "   ⚠️  详见日志: ${MRMR_LOG}" | tee -a "${STUDY_LOG}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_STUDIES="${FAILED_STUDIES} ${study}"
        continue
    fi
    
    # ----------------------------------------------------------------
    # Step 3: Stage2 PC算法精炼
    # ----------------------------------------------------------------
    log_step "$study" "Stage2精炼" "开始 (PC算法)"
    
    STAGE2_LOG="${LOG_DIR}/${study}_stage2.log"
    
    bash scripts/quick_stage2_refine.sh "${study}" \
        > "${STAGE2_LOG}" 2>&1
    
    if check_success; then
        log_step "$study" "Stage2精炼" "完成"
        
        # 显示生成的文件和基因数
        echo "   📂 生成文件:" | tee -a "${STUDY_LOG}"
        for fold_file in features/mrmr_stage2_${study}/fold_*.csv; do
            if [ -f "$fold_file" ]; then
                gene_count=$(tail -n +2 "$fold_file" | wc -l)
                echo "      $(basename $fold_file): ${gene_count} 个基因" | tee -a "${STUDY_LOG}"
            fi
        done
    else
        log_step "$study" "Stage2精炼" "失败"
        echo "   ⚠️  详见日志: ${STAGE2_LOG}" | tee -a "${STUDY_LOG}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_STUDIES="${FAILED_STUDIES} ${study}"
        continue
    fi
    
    # ----------------------------------------------------------------
    # Step 4: 消融实验
    # ----------------------------------------------------------------
    log_step "$study" "消融实验" "开始 (Gene/Text/Fusion)"
    
    ABLATION_LOG="${LOG_DIR}/${study}_ablation.log"
    
    bash scripts/run_ablation_study_mrmr_stage2.sh "${study}" \
        > "${ABLATION_LOG}" 2>&1
    
    if check_success; then
        log_step "$study" "消融实验" "完成"
        
        # 提取并显示结果
        RESULT_CSV="results/ablation_mrmr_stage2/${study}/final_comparison.csv"
        if [ -f "$RESULT_CSV" ]; then
            echo "   📊 消融实验结果:" | tee -a "${STUDY_LOG}"
            
            # 使用Python提取平均C-Index
            python3 << EOF | tee -a "${STUDY_LOG}"
import pandas as pd
import sys

try:
    df = pd.read_csv("${RESULT_CSV}")
    gene_mean = df['Gene_C_Index'].mean()
    text_mean = df['Text_C_Index'].mean()
    fusion_mean = df['Fusion_C_Index'].mean()
    
    print(f"      Gene Only:  {gene_mean:.4f}")
    print(f"      Text Only:  {text_mean:.4f}")
    print(f"      Fusion:     {fusion_mean:.4f}")
    
    if fusion_mean > gene_mean:
        improvement = ((fusion_mean - gene_mean) / gene_mean) * 100
        print(f"      提升: +{improvement:.2f}%")
except Exception as e:
    print(f"      ⚠️  无法解析结果: {e}")
    sys.exit(1)
EOF
        fi
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log_step "$study" "消融实验" "失败"
        echo "   ⚠️  详见日志: ${ABLATION_LOG}" | tee -a "${STUDY_LOG}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_STUDIES="${FAILED_STUDIES} ${study}"
        continue
    fi
    
    # ----------------------------------------------------------------
    # Step 5: 统计耗时
    # ----------------------------------------------------------------
    STUDY_END=$(date +%s)
    STUDY_DURATION=$((STUDY_END - STUDY_START))
    STUDY_MINUTES=$((STUDY_DURATION / 60))
    STUDY_SECONDS=$((STUDY_DURATION % 60))
    
    echo "   ⏱️  耗时: ${STUDY_MINUTES}分${STUDY_SECONDS}秒" | tee -a "${STUDY_LOG}"
    echo "" | tee -a "${MAIN_LOG}"
done

# ====================================================================
# 最终汇总
# ====================================================================

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "=============================================="
echo "🎉 批处理完成！"
echo "=============================================="
echo "📊 执行汇总:" | tee -a "${MAIN_LOG}"
echo "   ✅ 成功: ${SUCCESS_COUNT} 个癌症" | tee -a "${MAIN_LOG}"
echo "   ❌ 失败: ${FAIL_COUNT} 个癌症" | tee -a "${MAIN_LOG}"

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "   ⚠️  失败列表:${FAILED_STUDIES}" | tee -a "${MAIN_LOG}"
fi

echo "   ⏱️  总耗时: ${TOTAL_MINUTES}分${TOTAL_SECONDS}秒" | tee -a "${MAIN_LOG}"
echo "   📁 日志目录: ${LOG_DIR}" | tee -a "${MAIN_LOG}"
echo ""

# 汇总所有成功的结果
if [ ${SUCCESS_COUNT} -gt 0 ]; then
    echo "📊 所有成功癌症的结果汇总:" | tee -a "${MAIN_LOG}"
    echo "=============================================="
    
    for study in $STUDIES; do
        RESULT_CSV="results/ablation_mrmr_stage2/${study}/final_comparison.csv"
        if [ -f "$RESULT_CSV" ]; then
            echo ""
            echo "🧬 ${study^^}:" | tee -a "${MAIN_LOG}"
            
            python3 << EOF | tee -a "${MAIN_LOG}"
import pandas as pd
try:
    df = pd.read_csv("${RESULT_CSV}")
    gene_mean = df['Gene_C_Index'].mean()
    text_mean = df['Text_C_Index'].mean()
    fusion_mean = df['Fusion_C_Index'].mean()
    
    print(f"   Gene Only:  {gene_mean:.4f}")
    print(f"   Text Only:  {text_mean:.4f}")
    print(f"   Fusion:     {fusion_mean:.4f}")
    
    if fusion_mean > gene_mean:
        improvement = ((fusion_mean - gene_mean) / gene_mean) * 100
        print(f"   提升: +{improvement:.2f}%")
except:
    print("   ⚠️  无法读取结果")
EOF
        fi
    done
    
    echo ""
    echo "=============================================="
fi

echo ""
echo "✅ 全部完成！查看详细日志请访问: ${LOG_DIR}" | tee -a "${MAIN_LOG}"
echo ""

# 退出码
if [ ${FAIL_COUNT} -eq 0 ]; then
    exit 0
else
    exit 1
fi
